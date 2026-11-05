# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import threading
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torch.distributed import distributed_c10d

from torchft.hccl_recovery import HcclRecovery, HcclRecoveryError, RecoveryResult


@pytest.fixture
def deferred_timers(monkeypatch):
    timers = []

    class DeferredTimer:
        def __init__(self, interval, callback, args):
            self.interval = interval
            self.callback = lambda: callback(*args)
            self.cancelled = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr("torchft.hccl_recovery.threading.Timer", DeferredTimer)
    return timers


def _recovery(**overrides):
    values = dict(
        timeout=timedelta(seconds=1),
        recovery_timeout=timedelta(seconds=1),
        warmup_timeout=None,
        get_process_group=lambda: Mock(),
        clear_process_group=lambda pg: None,
    )
    values.update(overrides)
    return HcclRecovery(**values)


def test_only_unfinished_work_requests_recovery(monkeypatch, deferred_timers):
    requests = []
    monkeypatch.setattr(
        HcclRecovery,
        "_schedule_recovery",
        lambda self, token, generation: requests.append((token, generation)),
    )
    recovery = _recovery()
    recovery.prepare_group(0)
    recovery.group_created()

    recovery.wrap_work(Mock(), Mock(), object())
    generation = recovery.work_tracker.generation
    deferred_timers[-1].callback()

    assert requests == [(0, generation)]
    assert recovery.work_tracker.has_active_work()
    recovery.work_tracker.clear()


def test_timer_recovery_does_not_hold_boundary_lock(monkeypatch, deferred_timers):
    pg = Mock()
    pg._get_backend.return_value = Mock()
    stopped = threading.Event()
    recovery = _recovery(get_process_group=lambda: pg)
    recovery.prepare_group(0)
    recovery.group_created()
    recovery.wrap_work(Mock(), Mock(), object())
    generation = recovery.work_tracker.generation

    def stop(device):
        stopped.set()

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu=SimpleNamespace(
                stop_device=stop,
                restart_device=lambda device, **kwargs: None,
            )
        ),
    )
    monkeypatch.setattr(distributed_c10d, "_pg_map", {})
    monkeypatch.setattr(
        distributed_c10d,
        "_cleanup_process_group_global_state",
        lambda group: None,
        raising=False,
    )

    thread = threading.Thread(target=recovery._recover_device, args=(0, generation))
    thread.start()
    assert stopped.wait(1)

    # The training thread needs this lock to observe the stop.
    acquired = recovery.work_tracker.boundary_lock.acquire(timeout=1)
    try:
        assert acquired
        assert recovery.recover(RuntimeError("NPU FORCE STOP"))
    finally:
        if acquired:
            recovery.work_tracker.boundary_lock.release()

    thread.join(1)
    assert not thread.is_alive()
    assert recovery.num_device_recoveries == 1
    assert recovery.recovery_generation == 1

    # A monitor that observed the same peer-loss event before the timer
    # recovery must not stop/restart the device a second time.
    assert recovery.force_recover(recovery_generation=0) is RecoveryResult.COMPLETED
    assert recovery.num_device_recoveries == 1


def test_force_recover_does_not_block_on_boundary_lock(monkeypatch):
    pg = Mock()
    pg._get_backend.return_value = Mock()
    restarted = threading.Event()
    recovery = _recovery(get_process_group=lambda: pg)
    recovery.prepare_group(0)
    recovery.group_created()

    def restart(device, **kwargs):
        restarted.set()

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu=SimpleNamespace(
                stop_device=lambda device: None,
                restart_device=restart,
            )
        ),
    )
    monkeypatch.setattr(distributed_c10d, "_pg_map", {})
    monkeypatch.setattr(
        distributed_c10d,
        "_cleanup_process_group_global_state",
        lambda group: None,
        raising=False,
    )

    # Simulate the main thread holding the boundary lock while force_recover
    # runs; recovery must not wait for that lock to be released.
    acquired = recovery.work_tracker.boundary_lock.acquire(timeout=1)
    assert acquired
    try:
        thread = threading.Thread(target=recovery.force_recover)
        thread.start()
        assert recovery._recovery_requested.wait(1)
        assert recovery.recover(RuntimeError("NPU FORCE STOP"))
        thread.join(1)
        assert not thread.is_alive()
    finally:
        recovery.work_tracker.boundary_lock.release()

    assert restarted.is_set()
    assert recovery.num_device_recoveries == 1
    assert recovery.recovery_generation == 1


def test_abort_only_force_recover_is_unsupported(monkeypatch):
    from torchft.process_group import ProcessGroupHCCL

    monkeypatch.setenv("TORCHFT_HCCL_USE_ABORT", "true")
    pg = ProcessGroupHCCL()

    assert pg.force_recover() is RecoveryResult.UNSUPPORTED


def test_force_recover_stops_and_restarts_device(monkeypatch):
    calls = []
    stopped = threading.Event()
    backend = Mock()
    pg = Mock()
    pg._get_backend.return_value = backend

    recovery = _recovery(
        get_process_group=lambda: pg,
        clear_process_group=lambda group: calls.append(("clear", group)),
    )
    recovery.prepare_group(0)
    recovery.group_created()

    def stop(device):
        calls.append(("stop", device))
        stopped.set()

    def restart(device, **kwargs):
        calls.append(("restart", device, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu=SimpleNamespace(
                stop_device=stop,
                restart_device=restart,
            )
        ),
    )
    monkeypatch.setattr(distributed_c10d, "_pg_map", {})
    monkeypatch.setattr(
        distributed_c10d,
        "_cleanup_process_group_global_state",
        lambda group: calls.append(("unregister", group)),
        raising=False,
    )

    thread = threading.Thread(target=recovery.force_recover)
    thread.start()
    assert stopped.wait(1)
    assert recovery.recover(RuntimeError("NPU FORCE STOP"))
    thread.join(1)

    assert calls[0] == ("stop", 0)
    assert (
        "restart",
        0,
        {"rebuild_all_resources": True, "disable_tensor_unsafe_check": True},
    ) in calls
    assert ("unregister", pg) in calls
    assert ("clear", pg) in calls
    assert recovery.num_device_recoveries == 1
    assert recovery.recovery_generation == 1


def test_force_recover_returns_in_progress_when_not_ready():
    recovery = _recovery(get_process_group=lambda: None)

    assert recovery.force_recover() is RecoveryResult.IN_PROGRESS
    assert not recovery._recovery_requested.is_set()
    assert recovery._recovery_error is None


def test_force_recover_fails_after_not_ready_timeout(monkeypatch):
    recovery = _recovery(get_process_group=lambda: None)
    now = [1000.0]
    monkeypatch.setattr("torchft.hccl_recovery.time.monotonic", lambda: now[0])

    assert recovery.force_recover() is RecoveryResult.IN_PROGRESS
    now[0] += recovery.recovery_timeout.total_seconds() + 1
    assert recovery.force_recover() is RecoveryResult.FAILED
    assert isinstance(recovery._recovery_error, TimeoutError)
    assert recovery.force_recover() is RecoveryResult.FAILED


def test_prepare_group_resets_not_ready_timer(monkeypatch):
    recovery = _recovery(get_process_group=lambda: None)
    now = [1000.0]
    monkeypatch.setattr("torchft.hccl_recovery.time.monotonic", lambda: now[0])

    assert recovery.force_recover() is RecoveryResult.IN_PROGRESS
    assert recovery._not_ready_since == 1000.0

    # A process-group reinitialization starts a fresh not-ready window instead
    # of immediately failing based on the previous one.
    now[0] += recovery.recovery_timeout.total_seconds() + 1
    recovery.prepare_group(0)
    assert recovery._not_ready_since is None

    assert recovery.force_recover() is RecoveryResult.IN_PROGRESS
    assert recovery._not_ready_since == now[0]
    assert recovery._recovery_error is None


def test_force_recover_skips_completed_generation():
    recovery = _recovery(get_process_group=lambda: None)
    recovery._recovery_generation = 2
    recovery._not_ready_since = 1000.0

    assert recovery.force_recover(recovery_generation=1) is RecoveryResult.COMPLETED
    assert recovery._not_ready_since is None
    assert recovery.num_device_recoveries == 0
    assert recovery._recovery_error is None


def test_force_recover_failure_is_fatal_for_recover(monkeypatch):
    pg = Mock()
    pg._get_backend.return_value = Mock()
    recovery = _recovery(get_process_group=lambda: pg)
    recovery.prepare_group(0)
    recovery.group_created()

    def stop(device):
        raise RuntimeError("stop failed")

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu=SimpleNamespace(
                stop_device=stop,
                restart_device=lambda device, **kwargs: None,
            )
        ),
    )

    assert recovery.force_recover() is RecoveryResult.FAILED
    assert recovery._recovery_error is not None
    assert recovery._recovery_requested.is_set()

    with pytest.raises(RuntimeError):
        recovery.recover(RuntimeError("NPU FORCE STOP"))


def test_recovery_error_is_latched_before_device_ready(monkeypatch):
    pg = Mock()
    pg._get_backend.return_value = Mock()
    recovery = _recovery(get_process_group=lambda: pg)
    recovery.prepare_group(0)
    recovery.group_created()

    failure_started = threading.Event()
    allow_failure = threading.Event()
    ack_started = threading.Event()
    ready = threading.Event()
    error_at_ready = []

    class RecordingEvent:
        def set(self):
            error_at_ready.append(recovery._recovery_error)
            ready.set()

        def clear(self):
            ready.clear()

        def is_set(self):
            return ready.is_set()

        def wait(self, timeout=None):
            return ready.wait(timeout)

    recovery._device_ready = RecordingEvent()

    def stop(device):
        failure_started.set()
        allow_failure.wait(1)
        raise RuntimeError("stop failed")

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu=SimpleNamespace(
                stop_device=stop,
                restart_device=lambda device, **kwargs: None,
            )
        ),
    )

    results = []

    def acknowledge():
        ack_started.set()
        try:
            results.append(recovery.recover(RuntimeError("NPU FORCE STOP")))
        except Exception as error:
            results.append(error)

    recovery_thread = threading.Thread(target=recovery.force_recover)
    recovery_thread.start()
    assert failure_started.wait(1)

    ack_thread = threading.Thread(target=acknowledge)
    ack_thread.start()
    assert ack_started.wait(1)
    assert not ready.is_set()
    assert ack_thread.is_alive()

    allow_failure.set()
    recovery_thread.join(1)
    ack_thread.join(1)

    assert not recovery_thread.is_alive()
    assert not ack_thread.is_alive()
    # The ready event must only be set after the failure is latched.
    assert error_at_ready and error_at_ready[0] is not None
    assert isinstance(results[0], RuntimeError)
    assert recovery._recovery_error is not None


def test_reset_recovery_error_allows_external_restart():
    recovery = _recovery(get_process_group=lambda: None)
    recovery._recovery_error = RuntimeError("recovery failed")
    recovery._errored = RuntimeError("aborted")
    recovery._recovery_requested.set()
    recovery._device_ready.clear()

    recovery.reset_recovery_error()

    assert recovery._recovery_error is None
    assert recovery._errored is None
    assert not recovery._recovery_requested.is_set()
    assert recovery._device_ready.is_set()
    assert recovery.force_recover() is RecoveryResult.IN_PROGRESS


def test_force_recover_waits_for_ack_when_already_requested():
    recovery = _recovery()
    recovery._recovery_requested.set()

    assert recovery.force_recover() is RecoveryResult.IN_PROGRESS

    recovery._stop_observed.set()
    assert recovery.force_recover() is RecoveryResult.COMPLETED


def test_force_recover_waits_for_in_progress_failure():
    recovery = _recovery()
    lock_held = threading.Event()
    release_lock = threading.Event()
    calls = []

    def in_progress():
        recovery._recovery_lock.acquire()
        try:
            recovery._recovery_requested.set()
            lock_held.set()
            release_lock.wait(1)
            calls.append("failed")
            recovery._recovery_error = RuntimeError("in-progress recovery failed")
        finally:
            recovery._recovery_lock.release()

    thread = threading.Thread(target=in_progress)
    thread.start()
    assert lock_held.wait(1)

    def release():
        release_lock.set()

    releaser = threading.Thread(target=release)
    releaser.start()
    result = recovery.force_recover()
    thread.join(1)
    releaser.join(1)

    assert result is RecoveryResult.FAILED
    assert calls == ["failed"]


def test_recover_rejects_unrelated_error():
    recovery = _recovery()
    recovery._recovery_requested.set()

    assert not recovery.recover(RuntimeError("model error"))
    assert not recovery._stop_observed.is_set()


def test_apply_optimizer_rejects_while_recovering():
    recovery = _recovery()
    recovery._recovery_requested.set()

    with pytest.raises(HcclRecoveryError):
        recovery.apply_optimizer(lambda: None)


def test_wait_for_completion_accepts_untracked_work():
    work = Mock()
    work.wait.return_value = True
    recovery = _recovery()

    recovery.wait_for_completion(work)

    work.wait.assert_called_once_with()


def test_configure_aborts_collects_and_settles_before_recreate(monkeypatch):
    from torchft.process_group import ProcessGroupHCCL

    pg = ProcessGroupHCCL()
    pg._pg = Mock()
    calls = []

    monkeypatch.setenv("TORCHFT_HCCL_RECONFIGURE_SETTLE_S", "2")
    monkeypatch.setattr(pg, "abort", lambda errored: calls.append(("abort", errored)))
    monkeypatch.setattr("gc.collect", lambda: calls.append("gc"))
    monkeypatch.setattr(
        "torchft.process_group.time.sleep",
        lambda seconds: calls.append(("sleep", seconds)),
    )
    monkeypatch.setattr(
        "torchft.process_group.create_store_client",
        lambda store_addr, timeout: Mock(),
    )
    monkeypatch.setattr(
        pg,
        "_create_pg",
        lambda store, rank, world_size: calls.append("create") or Mock(),
    )

    pg.configure("store:1/prefix", "0", 0, 1)

    assert calls == [("abort", False), "gc", ("sleep", 2.0), "create"]
