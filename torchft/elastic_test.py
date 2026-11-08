# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import threading
from datetime import timedelta
from typing import Callable, List, Optional
from unittest import TestCase
from unittest.mock import create_autospec, MagicMock, patch

import torch
from torch.distributed import TCPStore
from torchft._torchft import QuorumResult
from torchft.elastic import (
    BatchPlan,
    DatasetIndexMapper,
    ElasticBatchPlanner,
    local_sample_range,
)
from torchft.manager import (
    Manager,
    MANAGER_ADDR_KEY,
    QuorumView,
    REPLICA_ID_KEY,
    WorldSizeMode,
)
from torchft.process_group import ManagedProcessGroup, ProcessGroup


class FakeManager:
    """The subset of Manager the planner uses, with a settable quorum.

    wait_plan() runs the registered plan function on the current quorum, as
    the quorum thread does at the membership point.
    """

    def __init__(self, min_replica_size: int = 1) -> None:
        self._min = min_replica_size
        self.P = 0
        self.qid = -1
        self.idx: Optional[int] = None
        self.step = 0
        self.plan_fn: Optional[Callable[[QuorumView], object]] = None

    def set(self, qid: int, P: int, idx: Optional[int], step: int = 0) -> None:
        self.qid, self.P, self.idx, self.step = qid, P, idx, step

    def min_replica_size(self) -> int:
        return self._min

    def set_plan_fn(self, fn: Optional[Callable[[QuorumView], object]]) -> None:
        self.plan_fn = fn

    def wait_plan(self) -> object:
        assert self.plan_fn is not None
        return self.plan_fn(
            QuorumView(
                quorum_id=self.qid,
                step=self.step,
                num_participants=self.P,
                participating_rank=self.idx,
            )
        )


def make_planner(
    manager: object,
    G: int = 48,
    micro: int = 2,
    D: int = 2,
    N: int = 4,
    max_ga: int = 100,
) -> ElasticBatchPlanner:
    # pyrefly: ignore [bad-argument-type]
    return ElasticBatchPlanner(manager, G, micro, D, N, max_ga)


def plans_for_all_replicas(
    G: int, micro: int, D: int, N: int, P: int, step: int, min_replica_size: int = 1
) -> List[BatchPlan]:
    plans = []
    for idx in range(P):
        m = FakeManager(min_replica_size)
        m.set(qid=7, P=P, idx=idx, step=step)
        plans.append(make_planner(m, G, micro, D, N).plan())
    return plans


class TestSharding(TestCase):
    def test_partition_exact(self) -> None:
        # covers r == 0 and r != 0 for every P up to U
        for G, micro, D in [(48, 2, 2), (60, 1, 3), (7, 1, 1), (64, 4, 1)]:
            U = G // (D * micro)
            N = U
            for P in range(1, U + 1):
                for step in (0, 1, 13):
                    plans = plans_for_all_replicas(G, micro, D, N, P, step)
                    ranges = [p.sample_range for p in plans]
                    self.assertTrue(all(r is not None for r in ranges))
                    covered = []
                    for r in ranges:
                        assert r is not None
                        covered.extend(range(*r))
                    msg = f"G={G} micro={micro} D={D} P={P} step={step}"
                    self.assertEqual(
                        covered, list(range(step * G, (step + 1) * G)), msg
                    )
                    self.assertEqual(
                        sum(p.grad_accum_steps * D * micro for p in plans), G, msg
                    )
                    gas = [p.grad_accum_steps for p in plans]
                    # every replica group sees the same U, and their counts sum to it
                    self.assertEqual({p.U for p in plans}, {U}, msg)
                    self.assertEqual(sum(gas), U, msg)
                    self.assertGreaterEqual(min(gas), 1, msg)
                    self.assertLessEqual(max(gas) - min(gas), 1, msg)
                    # the remainder goes to the lowest idx
                    self.assertEqual(gas, sorted(gas, reverse=True), msg)

    def test_non_participant(self) -> None:
        m = FakeManager()
        m.set(qid=3, P=2, idx=None, step=5)
        plan = make_planner(m).plan()
        self.assertEqual(plan.grad_accum_steps, 0)
        self.assertIsNone(plan.sample_range)
        self.assertIsNone(plan.idx)
        self.assertEqual((plan.quorum_id, plan.P), (3, 2))
        # a non-participant still scales its loss like the participants:
        # U = 48 // (2 * 2) with make_planner's defaults
        self.assertEqual(plan.U, 12)

    def test_ga_zero_iff_no_window(self) -> None:
        G, micro, D = 48, 2, 2
        U = G // (D * micro)
        for P in range(1, U + 1):
            for p in plans_for_all_replicas(G, micro, D, U, P, step=0):
                self.assertEqual(p.grad_accum_steps == 0, p.sample_range is None)


class TestValidation(TestCase):
    def assertConfigError(self, **kwargs: int) -> None:
        min_replica_size = kwargs.pop("min_replica_size", 1)
        with self.assertRaises(ValueError):
            make_planner(FakeManager(min_replica_size), **kwargs)

    def test_construction(self) -> None:
        # G not divisible by D * micro
        self.assertConfigError(G=50, micro=2, D=2)
        # N - 1 < min_replica_size
        self.assertConfigError(N=3, min_replica_size=3)
        # U < N: G=8, D*micro=4 -> U=2 < N=4
        self.assertConfigError(G=8, micro=2, D=2, N=4)
        # ceil(U / min) > max_grad_accum: U=12, min=2 -> 6 > 5
        self.assertConfigError(G=48, micro=2, D=2, N=4, max_ga=5, min_replica_size=2)
        # non-positive sizes
        self.assertConfigError(micro=0)
        self.assertConfigError(max_ga=0)

    def test_construction_boundary_ok(self) -> None:
        # ceil(12 / 2) == 6 == max_grad_accum
        make_planner(FakeManager(2), G=48, micro=2, D=2, N=4, max_ga=6)
        # U == N
        make_planner(FakeManager(1), G=16, micro=2, D=2, N=4)

    def test_no_quorum(self) -> None:
        m = FakeManager()
        planner = make_planner(m)
        with self.assertRaisesRegex(RuntimeError, "no quorum"):
            planner.plan()

    def test_more_replicas_than_rank_batches(self) -> None:
        m = FakeManager()
        planner = make_planner(m, G=48, micro=2, D=2, N=4)  # U = 12
        m.set(qid=1, P=13, idx=0)
        with self.assertRaisesRegex(RuntimeError, "more participating replicas"):
            planner.plan()

    def test_below_min_replica_size_does_not_raise(self) -> None:
        # P < min: still planned; should_commit blocks the commit. ga may
        # exceed max_grad_accum here: U=12, P=1 -> 12 > 6.
        m = FakeManager(min_replica_size=2)
        planner = make_planner(m, G=48, micro=2, D=2, N=4, max_ga=6)
        m.set(qid=1, P=1, idx=0)
        plan = planner.plan()
        self.assertEqual(plan.grad_accum_steps, 12)


class TestReplanDetection(TestCase):
    def test_plan_is_one_quorum(self) -> None:
        # P, idx, step and quorum_id all come from one QuorumView, so a plan
        # never mixes two quorums
        m = FakeManager()
        planner = make_planner(m)
        m.set(qid=1, P=2, idx=0, step=3)
        planner.plan()
        m.set(qid=2, P=3, idx=1, step=4)
        plan = planner.plan()
        self.assertEqual((plan.quorum_id, plan.P, plan.idx, plan.step), (2, 3, 1, 4))

    def test_registers_with_manager(self) -> None:
        m = FakeManager()
        planner = make_planner(m)
        self.assertEqual(m.plan_fn, planner.plan_from_view)

    def test_quorum_id_only_change_does_not_reassign(self) -> None:
        # Membership churn that leaves P and idx unchanged (new UUID in the
        # same slot, lagging replica joining) cannot change the assignment.
        m = FakeManager()
        planner = make_planner(m)
        m.set(qid=1, P=2, idx=0)
        planner.plan()
        m.set(qid=2, P=2, idx=0)
        with patch.object(planner, "_assign", wraps=planner._assign) as assign:
            plan = planner.plan()
        assign.assert_not_called()
        self.assertEqual(plan.quorum_id, 2)
        self.assertEqual(plan.grad_accum_steps, 6)

    def test_signal_scenarios(self) -> None:
        # Each scenario from the design's signal table, as a sequence of
        # (quorum_id, P, idx) for one observing replica, starting from a
        # steady state of (10, 3, 1), with the resulting ga for U=12.
        U_planner = {"G": 48, "micro": 2, "D": 2, "N": 6}  # U = 12
        start = (10, 3, 1)
        scenarios = {
            # join lagging (heal) -> catch up: P grows on the second round
            "scale_up_with_heal": [((11, 3, 1), 4), ((11, 4, 1), 3)],
            # slow replica evicted -> heal -> catch up
            "slow_replica_recover": [
                ((11, 2, 0), 6),
                ((12, 2, 0), 6),
                ((12, 3, 1), 4),
            ],
            "commit_failure_lag": [((11, 2, 1), 6), ((11, 3, 1), 4)],
            "scale_up_same_step": [((11, 4, 2), 3)],
            "scale_down": [((11, 2, 0), 6)],
            "replace_uuid": [
                ((11, 2, 1), 6),
                ((12, 2, 1), 6),
                ((12, 3, 2), 4),
            ],
            # FIXED_WITH_SPARES spare takes over / DYNAMIC one-in-one-out:
            # only idx moves
            "only_idx_changes": [((10, 3, 0), 4), ((10, 3, None), 0)],
            "all_commit_failed": [((11, 3, 1), 4)],
            "steady": [((10, 3, 1), 4), ((10, 3, 1), 4)],
        }
        for name, rounds in scenarios.items():
            m = FakeManager()
            planner = make_planner(m, **U_planner)
            m.set(*start)
            planner.plan()
            for step, (key, expect_ga) in enumerate(rounds, start=1):
                m.set(*key, step=step)
                plan = planner.plan()
                self.assertEqual(plan.grad_accum_steps, expect_ga, (name, key))
                self.assertEqual((plan.quorum_id, plan.P, plan.idx), key, name)

    def test_idx_only_change_moves_window(self) -> None:
        # Without idx in the key, the promoted replica would keep ga=0 and
        # its slot would go unread.
        m = FakeManager()
        planner = make_planner(m, G=48, micro=2, D=2, N=4)
        m.set(qid=5, P=2, idx=None, step=3)
        self.assertEqual(planner.plan().grad_accum_steps, 0)
        m.set(qid=5, P=2, idx=1, step=4)
        plan = planner.plan()
        self.assertEqual(plan.grad_accum_steps, 6)
        self.assertEqual(plan.sample_range, (4 * 48 + 24, 4 * 48 + 48))


class TestStepTranslation(TestCase):
    def test_window_advances_with_step(self) -> None:
        m = FakeManager()
        planner = make_planner(m, G=48, micro=2, D=2, N=4)
        m.set(qid=1, P=4, idx=2, step=0)
        first = planner.plan()
        m.step = 9
        later = planner.plan()
        assert first.sample_range is not None and later.sample_range is not None
        self.assertEqual(later.sample_range[0] - first.sample_range[0], 9 * 48)
        self.assertEqual(later.grad_accum_steps, first.grad_accum_steps)

    def test_retry_reads_same_window(self) -> None:
        # should_commit False -> step does not advance -> identical window
        m = FakeManager()
        planner = make_planner(m)
        m.set(qid=1, P=3, idx=1, step=4)
        a = planner.plan()
        m.qid = 2  # commit failure bumps quorum_id
        b = planner.plan()
        self.assertEqual(a.sample_range, b.sample_range)
        self.assertEqual(a.grad_accum_steps, b.grad_accum_steps)

    def test_cached_matches_fresh(self) -> None:
        m = FakeManager()
        planner = make_planner(m)
        for step, (P, idx) in enumerate(
            itertools.product(range(1, 5), [None, 0, 1, 2]), start=1
        ):
            if idx is not None and idx >= P:
                continue
            m.set(qid=step // 3, P=P, idx=idx, step=step)
            cached = planner.plan()
            # a separate manager: a second planner would replace the first
            # one's registration
            fresh_m = FakeManager()
            fresh_m.set(qid=step // 3, P=P, idx=idx, step=step)
            fresh = make_planner(fresh_m).plan()
            self.assertEqual(cached, fresh)


class TestLocalSampleRange(TestCase):
    def test_split_across_ranks(self) -> None:
        G, micro, D = 48, 2, 3  # U = 8
        for P in range(1, 9):
            for plan in plans_for_all_replicas(G, micro, D, 8, P, step=2):
                assert plan.sample_range is not None
                parts = [local_sample_range(plan, r, D) for r in range(D)]
                covered = []
                for part in parts:
                    assert part is not None
                    self.assertEqual(part[1] - part[0], plan.grad_accum_steps * micro)
                    covered.extend(range(*part))
                self.assertEqual(covered, list(range(*plan.sample_range)))

    def test_non_participant(self) -> None:
        m = FakeManager()
        m.set(qid=1, P=2, idx=None)
        plan = make_planner(m).plan()
        self.assertIsNone(local_sample_range(plan, 0, 2))


class TestDatasetIndexMapper(TestCase):
    def test_epoch_is_permutation(self) -> None:
        mapper = DatasetIndexMapper(dataset_len=17, seed=3)
        for epoch in range(3):
            got = [mapper(epoch * 17 + i) for i in range(17)]
            self.assertEqual(sorted(got), list(range(17)))

    def test_epochs_differ_and_cross_boundary(self) -> None:
        mapper = DatasetIndexMapper(dataset_len=50, seed=0)
        e0 = [mapper(i) for i in range(50)]
        e1 = [mapper(50 + i) for i in range(50)]
        self.assertNotEqual(e0, e1)
        # a window spanning the boundary reads the tail of e0 then the head of e1
        self.assertEqual([mapper(i) for i in range(45, 55)], e0[45:] + e1[:5])

    def test_deterministic_across_instances(self) -> None:
        a = DatasetIndexMapper(dataset_len=1000, seed=11)
        b = DatasetIndexMapper(dataset_len=1000, seed=11)
        # different access orders, same result
        idx = [5, 1999, 3, 1000, 2500]
        self.assertEqual([a(i) for i in idx], [b(i) for i in reversed(idx)][::-1])

    def test_no_shuffle(self) -> None:
        mapper = DatasetIndexMapper(dataset_len=10, shuffle=False)
        self.assertEqual([mapper(i) for i in range(8, 13)], [8, 9, 0, 1, 2])


def mock_should_commit(
    rank: int, step: int, should_commit: bool, timeout: timedelta
) -> bool:
    return should_commit


class TestWithManager(TestCase):
    """Runs the planner against the real Manager with a mocked quorum client."""

    store: TCPStore  # pyre-fixme[13]: never initialized

    def _create_manager(
        self,
        min_replica_size: int = 1,
        world_size_mode: WorldSizeMode = WorldSizeMode.DYNAMIC,
        use_async_quorum: bool = True,
    ) -> Manager:
        pg = create_autospec(ProcessGroup)
        # pyrefly: ignore [missing-attribute]
        pg.errored.return_value = None
        self.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        self.store.set(MANAGER_ADDR_KEY, "dummy")
        self.store.set(REPLICA_ID_KEY, "dummy_id")
        with patch(
            "os.environ",
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": self.store.port,
                "RANK": "1",
                "WORLD_SIZE": "2",
            },
        ):
            manager = Manager(
                pg=pg,
                min_replica_size=min_replica_size,
                load_state_dict=MagicMock(),
                state_dict=lambda: {},
                use_async_quorum=use_async_quorum,
                world_size_mode=world_size_mode,
                init_sync=False,
            )
        self.addCleanup(manager.shutdown, wait=False)
        return manager

    def _quorum(
        self,
        quorum_id: int,
        max_replica_rank: Optional[int],
        max_world_size: int,
        step: int = 0,
    ) -> QuorumResult:
        quorum = QuorumResult()
        quorum.quorum_id = quorum_id
        quorum.replica_rank = 0 if max_replica_rank is None else max_replica_rank
        quorum.replica_world_size = max_world_size
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = step
        quorum.max_replica_rank = max_replica_rank
        quorum.max_world_size = max_world_size
        quorum.heal = False
        return quorum

    def _healing_quorum(self, manager: Manager, client_mock: MagicMock) -> None:
        # this replica group (rank 1 of 2) is at step 0 and heals to step 20
        # from rank 0, the only one at max_step
        quorum = self._quorum(123, None, 1, step=20)
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.recover_src_replica_rank = 0
        quorum.heal = True
        client_mock()._quorum.return_value = quorum
        # stage a checkpoint for the transport to serve back
        manager._checkpoint_transport.send_checkpoint(
            dst_ranks=[],
            step=quorum.max_step,
            state_dict=manager._manager_state_dict(),
            timeout=timedelta(seconds=10),
        )
        client_mock()._checkpoint_metadata.return_value = (
            manager._checkpoint_transport.metadata()
        )

    def _record_plan_steps(self, manager: Manager) -> List[int]:
        # the Manager step at the moment the plan fn runs
        plan_fn = manager._plan_fn
        assert plan_fn is not None
        steps: List[int] = []

        def recording(view: QuorumView) -> object:
            steps.append(manager._step)
            return plan_fn(view)

        manager.set_plan_fn(recording)
        return steps

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_sync_heal_plans_after_healing(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=False)
        client_mock().should_commit = mock_should_commit
        planner = make_planner(manager, G=48, micro=2, D=2, N=4)  # U = 12
        steps = self._record_plan_steps(manager)
        self._healing_quorum(manager, client_mock)

        manager.start_quorum()
        plan = planner.plan()
        # healed before the plan: it trains step 20 with every replica group
        self.assertEqual(steps, [20])
        self.assertEqual((plan.step, plan.P, plan.idx), (20, 2, 1))
        self.assertEqual(plan.grad_accum_steps, 6)
        self.assertEqual(plan.sample_range, (20 * 48 + 24, 21 * 48))
        self.assertEqual(ManagedProcessGroup(manager).size(), 2)
        self.assertTrue(manager.is_participating())
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertTrue(manager.should_commit())
        self.assertEqual(manager.current_step(), 21)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_async_heal_plans_before_healing(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=True)
        planner = make_planner(manager, G=48, micro=2, D=2, N=4)
        steps = self._record_plan_steps(manager)
        self._healing_quorum(manager, client_mock)

        manager.start_quorum()
        plan = planner.plan()
        # outside the max-step view while healing: it reads nothing
        self.assertEqual((plan.step, plan.P, plan.idx), (0, 1, None))
        self.assertEqual(plan.grad_accum_steps, 0)
        manager.wait_quorum()
        self.assertEqual(steps, [0])
        self.assertEqual(manager.current_step(), 20)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_sync_reconfigure_error_still_plans(self, client_mock: MagicMock) -> None:
        # _async_quorum only reports a reconfigure error; wait_plan() must
        # still return, and the step does not commit
        manager = self._create_manager(use_async_quorum=False)
        client_mock().should_commit = mock_should_commit
        planner = make_planner(manager, G=48, micro=2, D=2, N=4)
        # pyrefly: ignore [missing-attribute]
        manager._pg.configure.side_effect = RuntimeError("configure failed")
        client_mock()._quorum.return_value = self._quorum(1, 1, 3)

        manager.start_quorum()
        plan = planner.plan()
        self.assertEqual((plan.step, plan.P, plan.idx), (0, 3, 1))
        self.assertIsNotNone(manager.errored())
        self.assertFalse(manager.should_commit())

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_sync_plan_error_does_not_fail_the_quorum(
        self, client_mock: MagicMock
    ) -> None:
        manager = self._create_manager(use_async_quorum=False)
        planner = make_planner(manager, G=48, micro=2, D=2, N=4)  # U = 12
        client_mock()._quorum.return_value = self._quorum(1, 0, 13)
        manager.start_quorum()
        with self.assertRaisesRegex(RuntimeError, "more participating replicas"):
            planner.plan()
        self.assertIsNone(manager.errored())
        self.assertEqual(ManagedProcessGroup(manager).size(), 13)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_accessors(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(min_replica_size=3)
        self.assertEqual(manager.min_replica_size(), 3)
        client_mock()._quorum.return_value = self._quorum(42, 0, 4)
        manager.start_quorum()
        manager.wait_quorum()
        self.assertEqual(manager.participating_rank(), 0)
        self.assertEqual(manager.num_participants(), 4)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_plan_before_start_quorum(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager)
        with self.assertRaisesRegex(RuntimeError, "no quorum"):
            planner.plan()

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_plan_reads_the_new_quorum(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        client_mock().should_commit = mock_should_commit
        planner = make_planner(manager, G=48, micro=2, D=2, N=4)  # U = 12

        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        plan = planner.plan()
        self.assertEqual((plan.quorum_id, plan.P, plan.idx), (1, 3, 1))
        self.assertEqual(plan.grad_accum_steps, 4)
        self.assertEqual(plan.sample_range, (16, 32))
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertTrue(manager.should_commit())

        # next quorum lands while plan() waits: P 3 -> 2, idx stays 1
        client_mock()._quorum.return_value = self._quorum(2, 1, 2, step=1)
        manager.start_quorum()
        plan = planner.plan()
        self.assertEqual((plan.quorum_id, plan.P, plan.idx), (2, 2, 1))
        self.assertEqual(plan.step, 1)
        self.assertEqual(plan.grad_accum_steps, 6)
        self.assertEqual(plan.sample_range, (48 + 24, 48 + 48))

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_fixed_with_spares(self, client_mock: MagicMock) -> None:
        # P is clamped to min_replica_size and ranks >= min become spares;
        # only idx tells a promoted spare to start reading.
        manager = self._create_manager(
            min_replica_size=2, world_size_mode=WorldSizeMode.FIXED_WITH_SPARES
        )
        planner = make_planner(manager, G=48, micro=2, D=2, N=4)

        client_mock()._quorum.return_value = self._quorum(1, 2, 3)
        manager.start_quorum()
        spare = planner.plan()
        self.assertEqual(spare.P, 2)
        self.assertIsNone(spare.idx)
        self.assertEqual(spare.grad_accum_steps, 0)

        # a lower-ranked replica leaves: same P after the clamp, idx 2 -> 1
        client_mock()._quorum.return_value = self._quorum(2, 1, 2)
        manager.start_quorum()
        promoted = planner.plan()
        self.assertEqual(promoted.P, 2)
        self.assertEqual(promoted.idx, 1)
        self.assertEqual(promoted.grad_accum_steps, 6)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_plan_does_not_wait_for_reconfigure(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager, G=48, micro=2, D=2, N=4)
        release = threading.Event()
        self.addCleanup(release.set)
        # pyrefly: ignore [missing-attribute]
        manager._pg.configure.side_effect = lambda *_: release.wait(10)

        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        plan = planner.plan()
        self.assertEqual((plan.quorum_id, plan.P, plan.idx), (1, 3, 1))
        # FSDP2 reads the replicate PG size on every micro-batch
        self.assertEqual(ManagedProcessGroup(manager).size(), 3)
        assert manager._quorum_future is not None
        self.assertFalse(manager._quorum_future.done())

        release.set()
        manager.wait_quorum()
        self.assertEqual(manager.num_participants(), 3)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_plan_error_does_not_fail_the_quorum(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager, G=48, micro=2, D=2, N=4)  # U = 12
        client_mock()._quorum.return_value = self._quorum(1, 0, 13)
        manager.start_quorum()
        with self.assertRaisesRegex(RuntimeError, "more participating replicas"):
            planner.plan()
        manager.wait_quorum()
        self.assertIsNone(manager.errored())
        self.assertEqual(ManagedProcessGroup(manager).size(), 13)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_error_reaches_plan(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager)
        client_mock()._quorum.side_effect = RuntimeError("quorum rpc failed")
        manager.start_quorum()
        with self.assertRaisesRegex(RuntimeError, "quorum rpc failed"):
            planner.plan()
        with self.assertRaisesRegex(RuntimeError, "quorum rpc failed"):
            ManagedProcessGroup(manager).size()

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_size_without_planner(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        with self.assertRaisesRegex(RuntimeError, "no quorum"):
            manager.wait_plan()
        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        self.assertEqual(ManagedProcessGroup(manager).size(), 3)
        assert manager._quorum_future is not None
        self.assertTrue(manager._quorum_future.done())

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_set_plan_fn_while_quorum_in_progress(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        planner = make_planner(manager, G=48, micro=2, D=2, N=4)
        release = threading.Event()
        self.addCleanup(release.set)
        # pyrefly: ignore [missing-attribute]
        manager._pg.configure.side_effect = lambda *_: release.wait(10)

        client_mock()._quorum.return_value = self._quorum(1, 1, 3)
        manager.start_quorum()
        # the plan is already out, but the round is still in reconfigure: a
        # change here would strand the next round's plan, so it is refused
        self.assertEqual(planner.plan().P, 3)
        with self.assertRaisesRegex(RuntimeError, "in progress"):
            manager.set_plan_fn(None)

        release.set()
        manager.wait_quorum()
        # between rounds it can be unregistered and the next round has no plan
        manager.set_plan_fn(None)
        manager.start_quorum()
        with self.assertRaisesRegex(RuntimeError, "no quorum"):
            manager.wait_plan()
        self.assertEqual(ManagedProcessGroup(manager).size(), 3)
