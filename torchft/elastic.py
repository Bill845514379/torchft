# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Elastic Batch
=============

This module splits a fixed global batch across a changing set of replica
groups.

Each step, :py:class:`ElasticBatchPlanner` reads the quorum result from the
:py:class:`~torchft.manager.Manager` and computes how many micro-batches this
replica group runs and which slice of the global sample stream it reads. The
plan is a pure function of ``(num_participants, participating_rank, step)``
so every replica group computes the same split without communicating.

The global batch size stays fixed when replicas join or leave: the
difference is absorbed by changing the number of gradient accumulation steps
on the surviving replicas.

Requirements:

* ``init_sync=False`` on the Manager.
* A healing replica group reads the window of the step it trains. With
  ``use_async_quorum=True`` (the default) it is outside the max-step view
  while it heals, so it reads no data. A synchronous quorum that allows
  healing heals it before the forward pass and computes the plan after
  healing, so it participates with the healed step.
* The dataset supports random access by index.
"""

from dataclasses import dataclass
from typing import cast, Dict, Optional, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchft.manager import Manager, QuorumView


# (P, idx): the assignment key. quorum_id does not affect the assignment, so
# a quorum bump that leaves both unchanged does not replan.
_PlanKey = Tuple[int, Optional[int]]


@dataclass(frozen=True)
class BatchPlan:
    """
    The data assignment for this replica group for one step.

    All sample counts are per replica group; ``sample_range`` indexes the
    global sample stream (see :py:class:`DatasetIndexMapper`), not the dataset.
    """

    step: int
    # diagnostics only; the assignment does not depend on it
    quorum_id: int
    # number of participating replica groups
    P: int
    # micro-batches per rank summed over all participating replica groups;
    # fixed while P changes. Scale the loss by U / P (not by
    # grad_accum_steps) so every sample weighs 1 / global_batch_size after
    # the cross-replica average, even when the groups' counts differ.
    U: int
    # micro-batches per rank this step; 0 iff this replica group reads no data
    grad_accum_steps: int
    # [start, end) in the global sample stream, None iff reads no data
    sample_range: Optional[Tuple[int, int]]
    # Manager.participating_rank(); None if outside the max-step view or a spare
    idx: Optional[int]


class ElasticBatchPlanner:
    """
    Computes the per-step :py:class:`BatchPlan` for this replica group.

    The constructor registers the planner with
    :py:meth:`Manager.set_plan_fn`, so the plan is computed on the quorum
    thread as soon as membership is known, before ProcessGroup
    reconfiguration and healing. A Manager holds one planner; construct it
    before the first ``manager.start_quorum()``. Call :py:meth:`plan` once
    per step, after ``manager.start_quorum()`` and before reading any data.

    Args:
        manager: the torchft Manager.
        global_batch_size: samples per step across all replica groups (G).
        micro_batch_size: samples per micro-batch per data-parallel rank.
        local_data_parallel_size: data-parallel ranks inside one replica group
            (D), i.e. the ranks that read distinct samples. Ranks that read
            identical data (tensor/context parallel) are not counted, so with
            context parallelism this is smaller than the fsdp dimension.
        num_replicas: the number of replica groups the job is deployed with
            (N).
        max_grad_accum: the largest number of micro-batches per rank per step
            that fits the step time budget.
    """

    def __init__(
        self,
        manager: "Manager",
        global_batch_size: int,
        micro_batch_size: int,
        local_data_parallel_size: int,
        num_replicas: int,
        max_grad_accum: int,
    ) -> None:
        self._manager = manager
        self._G = global_batch_size
        self._micro = micro_batch_size
        self._D = local_data_parallel_size
        self._N = num_replicas
        self._max_grad_accum = max_grad_accum
        self._min_replica_size: int = manager.min_replica_size()

        context: Dict[str, int] = {
            "global_batch_size": global_batch_size,
            "micro_batch_size": micro_batch_size,
            "local_data_parallel_size": local_data_parallel_size,
            "num_replicas": num_replicas,
            "max_grad_accum": max_grad_accum,
            "min_replica_size": self._min_replica_size,
        }
        for name, value in context.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1 ({context})")

        # samples in one rank-batch: one micro-batch on every rank of a replica
        self._rank_batch: int = self._D * self._micro
        if self._G % self._rank_batch != 0:
            raise ValueError(
                "global_batch_size must be divisible by "
                f"local_data_parallel_size * micro_batch_size ({context})"
            )
        self._U: int = self._G // self._rank_batch
        context["U"] = self._U

        if self._N - 1 < self._min_replica_size:
            raise ValueError(
                "num_replicas - 1 must be >= min_replica_size so that one "
                f"failure does not block commits ({context})"
            )
        if self._U < self._N:
            raise ValueError(
                "global rank-batches U must be >= num_replicas so every "
                f"participant gets at least one micro-batch ({context})"
            )
        # worst committed case is P == min_replica_size
        worst_ga = -(-self._U // self._min_replica_size)
        if worst_ga > self._max_grad_accum:
            raise ValueError(
                f"grad accumulation at P=min_replica_size would be {worst_ga} "
                f"> max_grad_accum ({context})"
            )

        self._cached_key: Optional[_PlanKey] = None
        # ga and the step-independent part of the sample offset for
        # _cached_key; written only on the quorum thread, by plan_from_view
        self._cached_ga: int = 0
        self._cached_local_offset: Optional[int] = None

        manager.set_plan_fn(self.plan_from_view)

    def plan_from_view(self, view: "QuorumView") -> BatchPlan:
        """
        Compute the plan for one quorum. The constructor registers this with
        :py:meth:`Manager.set_plan_fn`, so it runs on the quorum thread at the
        membership point; it only does arithmetic.

        Recomputes the assignment only when the replan key changed.
        """
        P = view.num_participants
        idx = view.participating_rank
        if P == 0:
            raise RuntimeError(f"no quorum (quorum_id={view.quorum_id})")
        if P > self._U:
            raise RuntimeError(
                f"more participating replicas ({P}) than global rank-batches "
                f"({self._U}); quorum_id={view.quorum_id}"
            )
        key = (P, idx)
        if key != self._cached_key:
            self._assign(key)
        return self._build(view, key)

    def _assign(self, key: _PlanKey) -> None:
        P, idx = key
        q, r = divmod(self._U, P)
        if idx is None:
            self._cached_ga = 0
            self._cached_local_offset = None
        else:
            self._cached_ga = q + (1 if idx < r else 0)
            self._cached_local_offset = (idx * q + min(idx, r)) * self._rank_batch
        self._cached_key = key

    def _build(self, view: "QuorumView", key: _PlanKey) -> BatchPlan:
        P, idx = key
        # the step the quorum was asked at: it does not advance on a failed
        # commit, so a retried step reads the same window
        step = view.step
        ga = self._cached_ga
        if self._cached_local_offset is None:
            sample_range = None
        else:
            start = step * self._G + self._cached_local_offset
            sample_range = (start, start + ga * self._rank_batch)
        return BatchPlan(
            step=step,
            quorum_id=view.quorum_id,
            P=P,
            U=self._U,
            grad_accum_steps=ga,
            sample_range=sample_range,
            idx=idx,
        )

    def plan(self) -> BatchPlan:
        """
        Get the plan for the current step.

        This waits for the in-flight quorum's membership, not for ProcessGroup
        reconfiguration or healing.
        """
        return cast(BatchPlan, self._manager.wait_plan())


def local_sample_range(
    plan: BatchPlan, local_dp_rank: int, local_dp_size: int
) -> Optional[Tuple[int, int]]:
    """
    Split a replica group's ``sample_range`` across its data-parallel ranks.

    Each rank gets ``grad_accum_steps * micro_batch_size`` consecutive global
    sample indices. Ranks that read identical data (tensor/context parallel)
    must pass the same ``local_dp_rank``.

    Returns:
        ``[start, end)`` in the global sample stream, or None if the replica
        group reads no data this step
    """
    if plan.sample_range is None:
        return None
    assert 0 <= local_dp_rank < local_dp_size, (local_dp_rank, local_dp_size)
    start, end = plan.sample_range
    per_rank, rem = divmod(end - start, local_dp_size)
    assert rem == 0, "replica sample window must be divisible by local_dp_size"
    rank_start = start + local_dp_rank * per_rank
    return (rank_start, rank_start + per_rank)


class DatasetIndexMapper:
    """
    Maps the global sample stream onto dataset indices.

    The stream is one random permutation of the dataset per epoch, laid end to
    end, so a window may span an epoch boundary. The permutation depends only
    on ``seed`` and the epoch, so every replica group gets the same mapping
    without communicating.

    Args:
        dataset_len: the number of samples in the dataset; must not change
            during training.
        seed: the base shuffle seed.
        shuffle: if False the stream is the dataset in order, repeated.
    """

    def __init__(self, dataset_len: int, seed: int = 0, shuffle: bool = True) -> None:
        if dataset_len < 1:
            raise ValueError("dataset_len must be >= 1")
        self._len = dataset_len
        self._seed = seed
        self._shuffle = shuffle
        self._epoch: Optional[int] = None
        self._perm: Optional[torch.Tensor] = None

    def _perm_for(self, epoch: int) -> torch.Tensor:
        if self._epoch != epoch or self._perm is None:
            generator = torch.Generator().manual_seed(self._seed + epoch)
            self._perm = torch.randperm(self._len, generator=generator)
            self._epoch = epoch
        return self._perm

    def __call__(self, global_index: int) -> int:
        if global_index < 0:
            raise ValueError("global_index must be >= 0")
        epoch, offset = divmod(global_index, self._len)
        if not self._shuffle:
            return offset
        return int(self._perm_for(epoch)[offset])
