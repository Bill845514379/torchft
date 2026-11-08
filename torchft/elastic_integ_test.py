# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end tests for ElasticBatchPlanner with a real lighthouse.

Each replica group reads only the samples its plan assigns and scales its loss
so the summed gradient equals the mean over the global batch regardless of how
it was split. The final weights must therefore match a single-process
reference run over the same sample stream; any lost, duplicated or misaligned
sample, or a retried step reading different data, shows up as a mismatch.
"""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple
from unittest import TestCase

import torch
from torch import nn, optim
from torch.distributed import ReduceOp
from torchft._torchft import LighthouseServer
from torchft.elastic import DatasetIndexMapper, ElasticBatchPlanner
from torchft.manager import Manager, WorldSizeMode
from torchft.manager_integ_test import EventInjector, EventInjectorEvent, Runner
from torchft.optim import OptimizerWrapper
from torchft.process_group import ProcessGroupGloo

INIT_LOCK: threading.Lock = threading.Lock()

DATASET_LEN = 37  # not a multiple of any G below, so windows cross epochs
LR = 0.05
WAIT_S = 60  # upper bound on any cross-replica wait; a hang fails the test
# Manager args for the synchronous-quorum variants
SYNC_QUORUM: Dict[str, Any] = {"use_async_quorum": False}


def make_dataset() -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(DATASET_LEN, 3, generator=generator)
    w = torch.tensor([[1.5], [-2.0], [0.5]])
    y = x @ w + 0.1 * torch.randn(DATASET_LEN, 1, generator=generator)
    return x, y


def make_model() -> nn.Module:
    with INIT_LOCK:
        # init_sync=False: every replica must start from identical weights
        torch.manual_seed(42)
        return nn.Linear(3, 1)


def reference_weights(G: int, steps: int) -> Dict[str, torch.Tensor]:
    x, y = make_dataset()
    mapper = DatasetIndexMapper(DATASET_LEN, seed=0)
    model = make_model()
    opt = optim.SGD(model.parameters(), lr=LR)
    for step in range(steps):
        idx = [mapper(g) for g in range(step * G, (step + 1) * G)]
        opt.zero_grad()
        loss = ((model(x[idx]) - y[idx]) ** 2).sum() / G
        loss.backward()
        opt.step()
    return model.state_dict()


def elastic_train_loop(
    rank: int,
    store_port: int,
    device: torch.device,
    runner: Runner,
    train_loop_args: Dict[str, Any],
) -> Dict[str, object]:
    """
    Per-replica options in train_loop_args (all optional unless noted):
      G, micro, steps, num_replicas, min_replica_size: required config
      error_at: {replica_id: step} fail the commit of that step once
      stay_down: after a crash, do not rejoin
      late_join: start only after replica 0 reaches ``signal_at``; with
        "ckpt" load replica 0's checkpoint first (same-step join), with
        "empty" start from step 0 and heal
      resume: checkpoint every replica loads before its first step
    Shared (one object for all replicas): "shared" with keys
      barrier, reached, release, ckpt, signal_at, pause_at_signal
    """
    G: int = train_loop_args["G"]
    micro: int = train_loop_args["micro"]
    steps: int = train_loop_args["steps"]
    num_replicas: int = train_loop_args["num_replicas"]
    min_replica_size: int = train_loop_args.get("min_replica_size", 2)
    shared: Dict[str, Any] = train_loop_args["shared"]
    late_join: Optional[str] = train_loop_args.get("late_join")

    # survives restarts: train_loop_args is the same dict across attempts
    log: List[Dict[str, object]] = train_loop_args.setdefault("log", [])
    attempt = train_loop_args.get("attempt", 0)
    train_loop_args["attempt"] = attempt + 1
    if attempt > 0 and train_loop_args.get("stay_down", False):
        # a permanently lost replica: do not rejoin after the crash
        return {"replica_id": runner.replica_id, "model": None, "log": log}
    if late_join is not None and attempt == 0:
        assert shared["reached"].wait(WAIT_S), "replica 0 never reached signal_at"

    with ExitStack() as stack:
        x, y = make_dataset()
        mapper = DatasetIndexMapper(DATASET_LEN, seed=0)
        model = make_model()
        inner = optim.SGD(model.parameters(), lr=LR)

        def load_state_dict(state_dict: Dict[str, Dict[str, object]]) -> None:
            model.load_state_dict(state_dict["model"])
            inner.load_state_dict(state_dict["optim"])

        def state_dict() -> Dict[str, Dict[str, object]]:
            return {"model": model.state_dict(), "optim": inner.state_dict()}

        pg = ProcessGroupGloo()
        manager = Manager(
            pg=pg,
            min_replica_size=min_replica_size,
            load_state_dict=load_state_dict,
            state_dict=state_dict,
            replica_id=str(runner.replica_id),
            store_addr="localhost",
            store_port=store_port,
            rank=rank,
            world_size=runner.world_size,
            lighthouse_addr=runner.lighthouse_address,
            init_sync=False,
            **runner.manager_args,
        )
        stack.callback(lambda: manager.shutdown(wait=False))

        def snapshot() -> Dict[str, Any]:
            return copy.deepcopy(
                {**state_dict(), "torchft": dict(manager.state_dict())}
            )

        def restore(ckpt: Dict[str, Any]) -> None:
            load_state_dict(ckpt)
            manager.load_state_dict(ckpt["torchft"])

        resume = train_loop_args.get("resume")
        if resume is not None and attempt == 0:
            restore(resume)
        if late_join == "ckpt":
            restore(shared["ckpt"])

        planner = ElasticBatchPlanner(
            manager,
            global_batch_size=G,
            micro_batch_size=micro,
            local_data_parallel_size=runner.world_size,
            num_replicas=num_replicas,
            max_grad_accum=G,
        )
        optimizer = OptimizerWrapper(manager, inner)
        if attempt == 0 and late_join is None:
            # Every initial Manager is heartbeating before anyone asks for a
            # quorum, so the lighthouse waits for all of them and the first
            # quorum always has every replica (which replica is a spare is
            # then fixed by replica_id order, not by start-up timing).
            shared["barrier"].wait()
        # replica_id -> step at which to fail the commit once
        error_at: Dict[int, int] = train_loop_args.get("error_at", {})
        first = True

        while manager.current_step() < steps:
            if (
                late_join is None
                and manager.current_step() == shared.get("signal_at")
                and not train_loop_args.get("signalled", False)
            ):
                train_loop_args["signalled"] = True
                if runner.replica_id == 0:
                    shared["ckpt"] = snapshot()
                    shared["reached"].set()
                if shared.get("pause_at_signal"):
                    # hold here, before asking for the next quorum, until
                    # the joiner's quorum request is already queued
                    assert shared["release"].wait(WAIT_S), "joiner never released"

            if first and late_join is not None:
                # The lighthouse forms a fast quorum as soon as all previous
                # members ask; give the joiner's request time to land first
                # so it is in the same quorum as them. A timer, since a
                # synchronous start_quorum blocks until that quorum forms.
                release = threading.Timer(1.0, shared["release"].set)
                release.start()
                stack.callback(release.cancel)
            first = False
            optimizer.zero_grad()  # start_quorum + clear grads, every step
            plan = planner.plan()

            if plan.grad_accum_steps == 0:
                # no data, but the collective must still be issued
                loss = model(x[:1]).sum() * 0.0
                loss.backward()
            else:
                assert plan.sample_range is not None
                start, end = plan.sample_range
                for i in range(plan.grad_accum_steps):
                    g = range(start + i * micro, start + (i + 1) * micro)
                    idx = [mapper(j) for j in g]
                    # weight by 1/G, not 1/ga: exact for uneven splits
                    loss = ((model(x[idx]) - y[idx]) ** 2).sum() / G
                    loss.backward()

            # Exactly one allreduce per step on every replica (a single bucket):
            # after an error torchft turns later allreduces of the step into
            # no-ops on that replica only, so several per step would desync.
            params = list(model.parameters())
            flat = torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None])
            assert flat.numel() == sum(p.numel() for p in params)
            manager.allreduce(flat, reduce_op=ReduceOp.SUM).wait()
            offset = 0
            for p in params:
                assert p.grad is not None
                n = p.numel()
                p.grad.copy_(flat.narrow(0, offset, n).view_as(p.grad))
                offset += n

            if error_at.get(runner.replica_id) == manager.current_step():
                del error_at[runner.replica_id]
                # what a failed allreduce does inside torchft
                manager.report_error(RuntimeError("injected commit failure"))
            entry = {
                "step": plan.step,  # provisional: stale on the healing replica
                "plan_step": plan.step,
                "quorum_id": plan.quorum_id,
                "P": plan.P,
                "idx": plan.idx,
                "ga": plan.grad_accum_steps,
                "range": plan.sample_range,
                "committed": False,
            }
            # logged before should_commit so a step that raises is still seen
            log.append(entry)
            committed = False
            try:
                committed = manager.should_commit()
            finally:
                # the plan is made at the membership point, but the healing
                # replica's step jump lands in the quorum thread's tail, so
                # its plan.step is the old step. Align every entry on the
                # committed step, which is unambiguous only after
                # should_commit.
                entry["step"] = manager.current_step() - (1 if committed else 0)
                entry["committed"] = committed
            if committed:
                inner.step()
            runner.event_injector.check(rank, manager.current_step())

        return {
            "replica_id": runner.replica_id,
            "model": {k: v.clone() for k, v in model.state_dict().items()},
            "log": log,
            "ckpt": snapshot(),
        }


class ElasticIntegTest(TestCase):
    def _run(
        self,
        num_replicas: int,
        G: int,
        micro: int,
        steps: int,
        event_injectors: Optional[List[EventInjector]] = None,
        min_replica_size: int = 2,
        manager_args: Optional[Dict[str, Any]] = None,
        error_at: Optional[Dict[int, int]] = None,
        stay_down: Tuple[int, ...] = (),
        late_join: Optional[Dict[int, str]] = None,
        signal_at: Optional[int] = None,
        pause_at_signal: bool = False,
        resume: Optional[Dict[str, Any]] = None,
        lighthouse_args: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, object]]:
        """Returns one result per replica, sorted by replica_id. A replica
        that raised has model None and the exception under "error"."""
        late_join = late_join or {}
        # join_timeout: give every started replica time to join the first
        # quorum (the 100ms default lets a slow thread miss it)
        lighthouse = LighthouseServer(
            bind="[::]:0",
            **{
                "min_replicas": min_replica_size,
                "join_timeout_ms": 2000,
                **(lighthouse_args or {}),
            },
        )
        self.addCleanup(lighthouse.shutdown)
        shared: Dict[str, Any] = {
            "barrier": threading.Barrier(num_replicas - len(late_join), timeout=WAIT_S),
            "reached": threading.Event(),
            "release": threading.Event(),
            "signal_at": signal_at,
            "pause_at_signal": pause_at_signal,
        }
        injectors = event_injectors or [EventInjector() for _ in range(num_replicas)]
        results: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            runs = []
            for replica_id, injector in zip(range(num_replicas), injectors):
                args: Dict[str, Any] = {
                    "G": G,
                    "micro": micro,
                    "steps": steps,
                    "num_replicas": num_replicas,
                    "min_replica_size": min_replica_size,
                    "error_at": dict(error_at or {}),
                    "stay_down": replica_id in stay_down,
                    "late_join": late_join.get(replica_id),
                    "resume": resume,
                    "shared": shared,
                }
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=injector,
                    train_loop=elastic_train_loop,
                    manager_args=manager_args or {},
                    train_loop_args=args,
                )
                runs.append((replica_id, args, executor.submit(runner.run_replica)))
            for replica_id, args, fut in runs:
                try:
                    # run_replica returns one result per rank; world_size == 1
                    results.append(fut.result()[0])
                except Exception as e:
                    results.append(
                        {
                            "replica_id": replica_id,
                            "model": None,
                            "log": args.get("log", []),
                            "error": e,
                        }
                    )
        return results

    def _check(
        self,
        results: List[Dict[str, object]],
        G: int,
        steps: int,
        start_step: int = 0,
    ) -> None:
        for r in results:
            self.assertNotIn("error", r, f"replica {r['replica_id']} raised")
        expected = reference_weights(G, steps)
        for r in results:
            if r["model"] is None:  # stayed down after a crash
                continue
            torch.testing.assert_close(
                r["model"], expected, msg=f"replica {r['replica_id']} diverged"
            )

        # Every committed step: the participants of that quorum tile the window.
        by_step: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
        for r in results:
            for entry in r["log"]:
                by_step.setdefault((entry["step"], entry["quorum_id"]), []).append(
                    entry
                )
        committed_steps = set()
        for (step, _), entries in by_step.items():
            if not any(e["committed"] for e in entries):
                continue
            for e in entries:
                if e["committed"] and e["idx"] is not None:
                    # a participant read the window of the step it committed
                    self.assertEqual(e["plan_step"], step, e)
            ranges = sorted(e["range"] for e in entries if e["range"] is not None)
            covered: List[int] = []
            for lo, hi in ranges:
                covered.extend(range(lo, hi))
            self.assertEqual(
                covered, list(range(step * G, (step + 1) * G)), f"step {step}"
            )
            committed_steps.add(step)
        self.assertEqual(committed_steps, set(range(start_step, steps)))

    @staticmethod
    def _collapse(log: List[Dict[str, object]]) -> List[Tuple[object, object]]:
        """(quorum_id, P) per step with consecutive repeats removed."""
        keys: List[Tuple[object, object]] = []
        for e in log:
            key = (e["quorum_id"], e["P"])
            if not keys or keys[-1] != key:
                keys.append(key)
        return keys

    def test_healthy(self) -> None:
        results = self._run(3, G=12, micro=2, steps=6)
        self._check(results, G=12, steps=6)
        for r in results:
            last = r["log"][-1]
            self.assertEqual((last["P"], last["ga"]), (3, 2))

    def test_uneven_split(self) -> None:
        # U = 10 over 3 replicas: ga = 4, 3, 3
        results = self._run(3, G=10, micro=1, steps=6)
        self._check(results, G=10, steps=6)
        gas = sorted(r["log"][-1]["ga"] for r in results)
        self.assertEqual(gas, [3, 3, 4])

    def test_replica_failure_and_rejoin(self) -> None:
        # Scale-down, then whole-replica replacement: replica 2 crashes after
        # step 2 and restarts with a fresh Manager (a new replica UUID), heals
        # and catches up. Survivors: P 3 -> 2 (replan) -> 3.
        injectors = [EventInjector(), EventInjector(), EventInjector().fail_at(0, 2)]
        results = self._run(3, G=12, micro=2, steps=8, event_injectors=injectors)
        self._check(results, G=12, steps=8)
        self.assertEqual(injectors[2].count[EventInjectorEvent.Failure], 1)
        survivor = results[0]["log"]
        survivor_P = [e["P"] for e in survivor]
        self.assertIn(2, survivor_P)
        self.assertEqual(survivor_P[-1], 3)
        # the 3 -> 2 step is a membership change: quorum_id moves with P
        down = next(i for i, e in enumerate(survivor) if e["P"] == 2)
        self.assertNotEqual(
            survivor[down]["quorum_id"], survivor[down - 1]["quorum_id"]
        )
        # with ga redistributed by divmod(6, 2)
        self.assertEqual(survivor[down]["ga"], 3)
        # the new UUID joins behind: P stays 2 until it catches up, and
        # catching up moves P without a quorum change
        up = next(i for i in range(down, len(survivor)) if survivor[i]["P"] == 3)
        self.assertTrue(all(e["P"] == 2 for e in survivor[down:up]))
        self.assertEqual(survivor[up]["quorum_id"], survivor[up - 1]["quorum_id"])
        # while healing the restarted replica read nothing
        rejoined = results[2]["log"]
        self.assertTrue(any(e["idx"] is None and e["ga"] == 0 for e in rejoined))

    def test_scale_up_late_join(self) -> None:
        # Replicas 0 and 1 train; replica 2 starts from scratch once replica 0
        # reaches step 3, joins behind, heals (reads nothing) and catches up.
        # 0 and 1 hold at step 3 until it asks for a quorum; otherwise they
        # can finish before it arrives.
        results = self._run(
            3,
            G=12,
            micro=2,
            steps=9,
            late_join={2: "empty"},
            signal_at=3,
            pause_at_signal=True,
        )
        self._check(results, G=12, steps=9)
        survivor_P = [P for _, P in self._collapse(results[0]["log"])]
        self.assertEqual(survivor_P[0], 2)
        self.assertEqual(survivor_P[-1], 3)
        joiner = results[2]["log"]
        self.assertTrue(any(e["idx"] is None and e["ga"] == 0 for e in joiner))
        self.assertGreaterEqual(joiner[0]["step"], 3)

    def test_replica_failure_and_rejoin_sync(self) -> None:
        # as test_replica_failure_and_rejoin with a synchronous quorum: the
        # restarted replica heals before its forward pass and participates in
        # the step it heals to, reading that step's window (_check: every
        # participant planned the step it committed). It is back in the first
        # quorum after its restart, so P may never drop to 2.
        injectors = [EventInjector(), EventInjector(), EventInjector().fail_at(0, 2)]
        results = self._run(
            3,
            G=12,
            micro=2,
            steps=8,
            event_injectors=injectors,
            manager_args=SYNC_QUORUM,
        )
        self._check(results, G=12, steps=8)
        self.assertEqual(injectors[2].count[EventInjectorEvent.Failure], 1)
        self.assertEqual(results[0]["log"][-1]["P"], 3)
        rejoined = results[2]["log"]
        self.assertTrue(all(e["idx"] is not None for e in rejoined), rejoined)

    def test_scale_up_late_join_sync(self) -> None:
        # as test_scale_up_late_join with a synchronous quorum: the joiner
        # starts from scratch, heals and participates in its first step
        results = self._run(
            3,
            G=12,
            micro=2,
            steps=9,
            late_join={2: "empty"},
            signal_at=3,
            pause_at_signal=True,
            manager_args=SYNC_QUORUM,
        )
        self._check(results, G=12, steps=9)
        joiner = results[2]["log"]
        self.assertGreaterEqual(joiner[0]["step"], 3)
        self.assertTrue(all(e["idx"] is not None for e in joiner), joiner)
        self.assertEqual(joiner[0]["P"], 3)

    def test_commit_failure_retry_sync(self) -> None:
        # as test_commit_failure_retry with a synchronous quorum: replica 1
        # heals to the others' step and splits the window with them at once
        results = self._run(
            3, G=12, micro=2, steps=7, error_at={1: 3}, manager_args=SYNC_QUORUM
        )
        self._check(results, G=12, steps=7)
        log1 = results[1]["log"]
        failed = next(i for i, e in enumerate(log1) if not e["committed"])
        heal = log1[failed + 1]
        self.assertEqual((heal["P"], heal["ga"]), (3, 2))
        self.assertEqual(heal["step"], log1[failed]["step"] + 1)
        self.assertTrue(heal["committed"])

    def test_same_step_join(self) -> None:
        # Replica 2 loads replica 0's checkpoint taken at the start of step 3
        # while 0 and 1 hold there, so all three ask for step 3's quorum at
        # the same step: it is a participant at once, without healing.
        results = self._run(
            3,
            G=12,
            micro=2,
            steps=7,
            late_join={2: "ckpt"},
            signal_at=3,
            pause_at_signal=True,
        )
        self._check(results, G=12, steps=7)
        joiner = results[2]["log"]
        self.assertEqual(joiner[0]["step"], 3)
        self.assertTrue(all(e["idx"] is not None for e in joiner), joiner)
        self.assertEqual(joiner[0]["P"], 3)
        survivor = {e["step"]: e for e in results[0]["log"]}
        self.assertEqual(survivor[2]["P"], 2)
        self.assertEqual(survivor[3]["P"], 3)
        self.assertNotEqual(survivor[3]["quorum_id"], survivor[2]["quorum_id"])

    def test_commit_failure_retry(self) -> None:
        # replica 1 fails to commit step 3: it falls one step behind and
        # heals; the others commit with the full batch
        results = self._run(3, G=12, micro=2, steps=7, error_at={1: 3})
        self._check(results, G=12, steps=7)
        log1 = results[1]["log"]
        failed = next(i for i, e in enumerate(log1) if not e["committed"])
        # the others moved on, so it heals to their step and reads nothing;
        # meanwhile the other two split the window (ga 3 each)
        heal = log1[failed + 1]
        self.assertEqual((heal["idx"], heal["ga"]), (None, 0))
        self.assertEqual(heal["step"], log1[failed]["step"] + 1)
        self.assertEqual(results[0]["log"][failed + 1]["ga"], 3)

    def test_commit_failure_all_replicas(self) -> None:
        # every replica fails to commit step 3: nobody advances, and the
        # retry reads exactly the same windows
        results = self._run(3, G=12, micro=2, steps=7, error_at={0: 3, 1: 3, 2: 3})
        self._check(results, G=12, steps=7)
        for r in results:
            log = r["log"]
            failed = next(i for i, e in enumerate(log) if not e["committed"])
            retry = log[failed + 1]
            self.assertTrue(retry["committed"])
            self.assertEqual(
                (retry["step"], retry["range"]),
                (log[failed]["step"], log[failed]["range"]),
            )

    def test_fixed_with_spares(self) -> None:
        # 3 replicas, 2 active. Replica 0 is lost for good after step 2; spare
        # 2 (last in replica_id order) must take a slot -- for it only idx
        # changes -- and read data. Replica 0 stays down: if it rejoined it
        # would sort ahead of 2 and win its slot back, so the promotion would
        # depend on timing.
        injectors = [EventInjector().fail_at(0, 2), EventInjector(), EventInjector()]
        results = self._run(
            3,
            G=12,
            micro=2,
            steps=8,
            event_injectors=injectors,
            min_replica_size=2,
            manager_args={"world_size_mode": WorldSizeMode.FIXED_WITH_SPARES},
            stay_down=(0,),
        )
        self._check(results, G=12, steps=8)
        for r in results:
            self.assertTrue(all(e["P"] <= 2 for e in r["log"]))
        logs = {r["replica_id"]: r["log"] for r in results}
        spare = results[2]["log"]
        self.assertTrue(any(e["idx"] is None for e in spare), logs)
        self.assertTrue(
            any(e["idx"] is not None and e["committed"] for e in spare), logs
        )

    def test_fixed_with_spares_slot_reclaimed(self) -> None:
        # Replica 0 fails to commit step 3 and falls behind: spare 2 covers
        # for it, then 0 catches up and takes its slot back. On that round
        # the member list and P (clamped to 2) are unchanged and only idx
        # moves (§1.1 "不变/不变/变"): replica 1 goes from slot 0 to slot 1
        # and must replan, or slot 1 goes unread.
        results = self._run(
            3,
            G=12,
            micro=2,
            steps=8,
            min_replica_size=2,
            manager_args={"world_size_mode": WorldSizeMode.FIXED_WITH_SPARES},
            error_at={0: 3},
        )
        self._check(results, G=12, steps=8)
        log1 = results[1]["log"]
        only_idx = [
            (a, b)
            for a, b in zip(log1, log1[1:])
            if (a["quorum_id"], a["P"]) == (b["quorum_id"], b["P"])
            and a["idx"] != b["idx"]
        ]
        self.assertTrue(only_idx, {r["replica_id"]: r["log"] for r in results})

    def test_below_min_replicas(self) -> None:
        # The lighthouse accepts a 1-replica quorum but the Manager needs 2:
        # after replica 1 is lost the survivor keeps planning (P=1 < min is
        # not an error), never commits, and exits after max_retries.
        # (2 replicas with min 2 is already a config error: N-1 < min)
        injectors = [
            EventInjector(),
            EventInjector().fail_at(0, 2),
            EventInjector().fail_at(0, 2),
        ]
        results = self._run(
            3,
            G=12,
            micro=2,
            steps=8,
            event_injectors=injectors,
            min_replica_size=2,
            manager_args={"max_retries": 2},
            stay_down=(1, 2),
            lighthouse_args={"min_replicas": 1, "heartbeat_timeout_ms": 500},
        )
        survivor = results[0]
        self.assertIn("max_retries", str(survivor.get("error")), survivor["log"])
        low = [e for e in survivor["log"] if e["P"] == 1]
        # max_retries=2: the third consecutive failure raises
        self.assertEqual(len(low), 3, survivor["log"])
        for e in low:
            self.assertFalse(e["committed"])
            self.assertEqual((e["step"], e["idx"], e["ga"]), (2, 0, 6))
        self.assertTrue(all(e["committed"] for e in survivor["log"] if e["P"] == 3))

    def test_job_restart_from_checkpoint(self) -> None:
        # Stop the whole job after step 4, restart every replica from
        # replica 0's checkpoint: training resumes at window 4*G and ends
        # where an uninterrupted run would.
        first = self._run(3, G=12, micro=2, steps=4)
        self._check(first, G=12, steps=4)
        ckpt = first[0]["ckpt"]
        self.assertEqual(ckpt["torchft"]["step"], 4)
        second = self._run(3, G=12, micro=2, steps=8, resume=ckpt)
        self._check(second, G=12, steps=8, start_step=4)
        for r in second:
            self.assertEqual(r["log"][0]["step"], 4)
            self.assertTrue(all(e["idx"] is not None for e in r["log"]))
