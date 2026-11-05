# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HCCL device recovery for TorchFT.

This module implements the scheme validated on Ascend NPU:

* a step-level side thread detects a hang outside the tracked TorchFT PG;
* ``stop_device`` interrupts the in-flight collective;
* ``restart_device`` rebuilds device resources;
* every remaining NPU PG deletes its stale TCP store key and aborts its HCCL
  communicator so the next collective re-creates the link from fresh root info;
* the interrupted attempt is discarded and the next step starts a new quorum.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, cast

import torch

from torchft.process_group import _WorkAcceleratorTimeout

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup, Work

logger = logging.getLogger(__name__)


class HcclRecoveryError(RuntimeError):
    """The current HCCL attempt must end before device recovery can complete."""


class RecoveryResult(Enum):
    """Outcome of a forced device recovery attempt."""

    COMPLETED = auto()
    IN_PROGRESS = auto()
    FAILED = auto()
    UNSUPPORTED = auto()


class HcclWorkTracker:
    """Track unfinished HCCL work without letting old timers affect a new step."""

    def __init__(self, on_timeout: Callable[[int | None, int], None]) -> None:
        self.boundary_lock = threading.RLock()
        self._lock = threading.Lock()
        self._on_timeout = on_timeout
        self._generation = 0
        self._next_token = 0
        self._active: dict[int, tuple[threading.Timer, Any]] = {}

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def track(self, timeout: timedelta) -> int:
        self.retire_completed()
        with self._lock:
            token = self._next_token
            self._next_token += 1
            generation = self._generation
            timer = threading.Timer(
                timeout.total_seconds(), self._expire, args=(token, generation)
            )
            timer.daemon = True
            self._active[token] = (timer, None)
            timer.start()
            return token

    def record_completion(self, token: int) -> None:
        import torch_npu

        event = torch_npu.npu.Event()
        event.record()
        with self._lock:
            if token in self._active:
                timer, _ = self._active[token]
                self._active[token] = (timer, event)

    def complete(self, token: int) -> None:
        with self._lock:
            entry = self._active.pop(token, None)
        if entry is not None:
            entry[0].cancel()

    def retire_completed(self) -> None:
        with self._lock:
            entries = tuple(self._active.items())
        for token, (_, event) in entries:
            if event is not None:
                try:
                    if event.query():
                        self.complete(token)
                except RuntimeError:
                    # A stopped device cannot confirm that the work completed.
                    pass

    def has_active_work(self, token: int | None = None) -> bool:
        self.retire_completed()
        with self._lock:
            return bool(self._active) if token is None else token in self._active

    def clear(self) -> None:
        # Only _lock is needed: generation and the active map are fully protected
        # by it. Avoid boundary_lock so recovery can clear stale work while the
        # training thread is blocked holding the boundary lock.
        with self._lock:
            timers = [timer for timer, _ in self._active.values()]
            self._active.clear()
            self._generation += 1
        for timer in timers:
            timer.cancel()

    def _expire(self, token: int, generation: int) -> None:
        self.retire_completed()
        with self._lock:
            if generation != self._generation or token not in self._active:
                return
        self._on_timeout(token, generation)


class TrackedHcclWork(_WorkAcceleratorTimeout):
    _work: Any

    def __init__(
        self, pg: Any, work: Any, timeout: timedelta, *, tracker: HcclWorkTracker
    ) -> None:
        super().__init__(pg, work, timeout)
        self._tracker = tracker
        self._token = self._tracker.track(timeout)

    @classmethod
    @contextmanager
    def _stream_timeout(cls, pg: Any, timeout: timedelta):
        # The owner timer is generation-scoped; the upstream timer is not.
        yield

    def wait(self, timeout: timedelta | None = None) -> bool:
        completed = super().wait(timeout)
        if completed:
            self._tracker.record_completion(self._token)
        return completed

    def is_completed(self) -> bool:
        completed = self._work.is_completed()
        if completed:
            # WorkHCCL.is_completed() queries this collective's HCCL end event,
            # so it can retire the timer without synchronizing the compute stream.
            self._tracker.complete(self._token)
        return completed

    def get_future(self) -> torch.futures.Future[object]:
        future = self._work.get_future()
        tracker, token = self._tracker, self._token

        def record_completion(done: torch.futures.Future[object]) -> None:
            done.wait()
            tracker.record_completion(token)

        future.add_done_callback(record_completion)
        return future


class HcclRecovery:
    """Coordinate HCCL device recovery with work and optimizer boundaries."""

    def __init__(
        self,
        timeout: timedelta,
        *,
        recovery_timeout: timedelta,
        warmup_timeout: timedelta | None,
        get_process_group: Callable[[], ProcessGroup | None],
        clear_process_group: Callable[[ProcessGroup], None],
    ) -> None:
        if recovery_timeout.total_seconds() <= 0:
            raise ValueError("recovery_timeout must be positive")
        if warmup_timeout is not None and warmup_timeout.total_seconds() <= 0:
            raise ValueError("warmup_timeout must be positive")
        self._timeout = timeout
        self._warmup_timeout = max(timeout, warmup_timeout or timeout)
        self._warmup_pending = False
        self.recovery_timeout = recovery_timeout
        self._get_process_group = get_process_group
        self._clear_process_group = clear_process_group
        self.work_tracker = HcclWorkTracker(self._schedule_recovery)
        self._recovery_lock = threading.Lock()
        self._recovery_requested = threading.Event()
        self._stop_observed = threading.Event()
        self._device_ready = threading.Event()
        self._device_ready.set()
        self._recovery_error: BaseException | None = None
        self._errored: Exception | None = None
        self._device_index: int | None = None
        self._recovery_generation = 0
        self._not_ready_since: float | None = None
        self.num_device_recoveries = 0
        self.optimizer_update_completed = False

    @property
    def watchdog_timeout(self) -> timedelta:
        # Leave time for Python recovery before the fatal HCCL watchdog.
        return self._warmup_timeout + self.recovery_timeout + timedelta(seconds=30)

    @property
    def recovery_generation(self) -> int:
        """Number of successful device recoveries completed by this instance."""
        with self._recovery_lock:
            return self._recovery_generation

    def errored(self) -> Exception | None:
        return self._errored

    def prepare_group(self, device_index: int) -> None:
        self._raise_if_recovering()
        self._device_index = device_index
        self._not_ready_since = None

    def group_created(self) -> None:
        self.work_tracker.clear()
        self._warmup_pending = True
        self._errored = None

    @property
    def _collective_timeout(self) -> timedelta:
        return self._warmup_timeout if self._warmup_pending else self._timeout

    @contextmanager
    def track_collective(self):
        self._raise_if_recovering()
        # HCCL can block before returning a Work, including initial connection.
        token = self.work_tracker.track(self._collective_timeout)
        try:
            yield
        finally:
            self.work_tracker.complete(token)

    def wrap_work(self, pg: ProcessGroup, work: Work, opts: object) -> Work:
        timeout = self._collective_timeout
        requested = getattr(opts, "timeout", None)
        if isinstance(requested, timedelta) and timedelta(0) < requested < timeout:
            timeout = requested
        return TrackedHcclWork(pg, work, timeout, tracker=self.work_tracker)

    def wait_for_completion(self, work: Work) -> None:
        """Wait for one HCCL work item without synchronizing the compute stream."""
        tracked_work = getattr(work, "_work", None)
        if not isinstance(tracked_work, TrackedHcclWork):
            if not work.wait():
                raise RuntimeError("HCCL work did not complete")
            return
        while True:
            with self.work_tracker.boundary_lock:
                self._raise_if_recovering()
                try:
                    if tracked_work.is_completed():
                        break
                except RuntimeError:
                    self._raise_if_recovering()
                    raise
            time.sleep(0.001)
        with self.work_tracker.boundary_lock:
            self._raise_if_recovering()
        # WorkHCCL.wait() releases allocator-safety state. Calling it only
        # after this work's end event completed avoids waiting on unrelated
        # compute while retaining the required record-stream cleanup.
        if not work.wait():
            raise RuntimeError("HCCL work did not complete")

    @contextmanager
    def release_group(self, *, errored: bool):
        with self.work_tracker.boundary_lock:
            self._raise_if_recovering()
            if errored:
                self._errored = RuntimeError("aborted")
            pg = self._get_process_group()
            if pg is None:
                yield None
                return
            if self.work_tracker.has_active_work():
                if errored:
                    self._schedule_recovery(None, self.work_tracker.generation)
                    yield None
                    return
                raise RuntimeError("Cannot release an HCCL group with unfinished work")
            try:
                yield pg
            finally:
                self._clear_process_group(pg)
                self.work_tracker.clear()

    def _raise_if_recovering(self) -> None:
        if self._recovery_error is not None:
            raise RuntimeError(
                "TorchFT NPU device recovery failed; restart the training job"
            ) from self._recovery_error
        if self._recovery_requested.is_set():
            raise HcclRecoveryError(
                "The interrupted HCCL attempt must finish before retrying"
            )

    def _schedule_recovery(self, token: int | None, generation: int) -> None:
        # Never block the timer callback on communicator destruction.
        thread = threading.Thread(
            target=self._recover_device, args=(token, generation), daemon=True
        )
        thread.start()

    def _npu_process_group(self) -> ProcessGroup | None:
        pg = self._get_process_group()
        if pg is not None:
            return pg
        from torch.distributed.distributed_c10d import _pg_map

        npu_device = torch.device("npu")
        for candidate in tuple(_pg_map):
            if npu_device in candidate._device_types:
                return candidate
        return None

    def _begin_recovery(self, error_message: str) -> None:
        """Mark the current attempt as interrupted."""
        self._device_ready.clear()
        self._stop_observed.clear()
        self._recovery_requested.set()
        self._errored = HcclRecoveryError(error_message)

    def _perform_recovery(self, pg: ProcessGroup, reason: str) -> None:
        """Stop/restart the device and rebuild communicators."""
        try:
            self._perform_recovery_locked(pg, reason)
        except BaseException as error:
            # Latch the failure before waking ``recover()`` so it cannot observe a
            # ready device without the corresponding error.
            self._recovery_error = error
            raise
        finally:
            self._device_ready.set()

    def _perform_recovery_locked(self, pg: ProcessGroup, reason: str) -> None:
        assert self._device_index is not None
        backend = cast("Any", pg._get_backend(torch.device("npu")))
        import torch_npu

        logger.warning("Stopping NPU %d %s", self._device_index, reason)
        result = torch_npu.npu.stop_device(self._device_index)
        if result not in (None, 0):
            raise RuntimeError(f"NPU stop_device failed: {result}")
        backend.abort_hccl_comm("reinit")
        if not self._stop_observed.wait(self.recovery_timeout.total_seconds()):
            raise TimeoutError(
                "Training thread did not observe the interrupted NPU attempt"
            )
        from torch.distributed.distributed_c10d import (
            _cleanup_process_group_global_state,
            _pg_map,
        )

        backend.clear_workmeta_list()
        _cleanup_process_group_global_state(pg)
        self._clear_process_group(pg)
        self.work_tracker.clear()
        torch_npu.npu.restart_device(
            self._device_index,
            rebuild_all_resources=True,
            disable_tensor_unsafe_check=True,
        )
        npu_device = torch.device("npu")
        for remaining_pg in tuple(_pg_map):
            if npu_device in remaining_pg._device_types:
                remaining_backend = cast("Any", remaining_pg._get_backend(npu_device))
                remaining_backend._delete_tcpstore_key()
                remaining_backend.abort_hccl_comm("reinit")
        self.num_device_recoveries += 1
        self._recovery_generation += 1
        self._not_ready_since = None
        logger.warning("Restarted NPU %d %s", self._device_index, reason)

    def _recover_device(self, token: int | None, generation: int) -> None:
        self._recovery_lock.acquire()
        try:
            with self.work_tracker.boundary_lock:
                pg = self._get_process_group()
                if generation != self.work_tracker.generation or pg is None:
                    return
                if (
                    not self.work_tracker.has_active_work(token)
                    or self._recovery_requested.is_set()
                ):
                    return
                self._begin_recovery(
                    "HCCL communication timed out during an uncommitted attempt"
                )
            # The training thread needs boundary_lock to observe the stop, so run
            # the actual recovery outside of it.
            self._perform_recovery(
                pg,
                "after unfinished TorchFT HCCL communication",
            )
        except BaseException as error:
            self._recovery_error = error
            logger.exception("TorchFT HCCL device recovery failed")
        finally:
            self._recovery_lock.release()

    def force_recover(self, recovery_generation: int | None = None) -> RecoveryResult:
        """Stop/restart the device when the training thread is stuck outside
        the tracked TorchFT collectives (for example in an FSDP/mesh all-reduce).
        """
        self._recovery_lock.acquire()
        try:
            if self._recovery_error is not None:
                return RecoveryResult.FAILED
            if (recovery_generation is not None
                and self._recovery_generation > recovery_generation):
                self._not_ready_since = None
                return RecoveryResult.COMPLETED
            if self._recovery_requested.is_set():
                # The lock guarantees the timer-driven recovery has finished.
                # Until the training thread acknowledges the stop, the interrupted
                # attempt may still be running, so the epoch must not be consumed.
                if self._stop_observed.is_set():
                    return RecoveryResult.COMPLETED
                return RecoveryResult.IN_PROGRESS
            pg = self._npu_process_group()
            if pg is None or self._device_index is None:
                now = time.monotonic()
                if self._not_ready_since is None:
                    self._not_ready_since = now
                elif (
                    now - self._not_ready_since > self.recovery_timeout.total_seconds()
                ):
                    self._recovery_error = TimeoutError(
                        "NPU process group was not ready for peer-loss recovery "
                        "within recovery_timeout"
                    )
                    logger.error(
                        "TorchFT HCCL forced device recovery gave up waiting "
                        "for the NPU process group"
                    )
                    return RecoveryResult.FAILED
                return RecoveryResult.IN_PROGRESS
            self._not_ready_since = None
            self._begin_recovery("HCCL communication hang during a training step")
            self._perform_recovery(pg, "after a stuck training step")
            return RecoveryResult.COMPLETED
        except BaseException as error:
            self._recovery_error = error
            logger.exception("TorchFT HCCL forced device recovery failed")
            return RecoveryResult.FAILED
        finally:
            self._recovery_lock.release()

    def reset_recovery_error(self) -> None:
        """Clear a terminal recovery failure."""
        with self._recovery_lock:
            self._recovery_error = None
            self._errored = None
            self._recovery_requested.clear()
            self._stop_observed.clear()
            self._device_ready.set()
            self._not_ready_since = None

    def recover(self, error: Exception) -> bool:
        """Acknowledge the expected stop, then wait before accessing device state."""
        if not self._recovery_requested.is_set():
            return False
        if (not isinstance(error, HcclRecoveryError)
            and "FORCE STOP" not in str(error).upper()):
            return False
        self._stop_observed.set()
        if not self._device_ready.wait(self.recovery_timeout.total_seconds()):
            raise TimeoutError(
                "Timed out waiting for TorchFT NPU device recovery"
            ) from error
        if self._recovery_error is not None:
            raise RuntimeError(
                "TorchFT NPU device recovery failed; restart the training job"
            ) from self._recovery_error
        self._recovery_requested.clear()
        self._errored = None
        return True

    def begin_attempt(self) -> None:
        self.optimizer_update_completed = False

    def apply_optimizer(self, update: Callable[[], None]) -> None:
        """Exclude late timeouts until the accepted local update has completed."""
        with self.work_tracker.boundary_lock:
            self._raise_if_recovering()
            # Manager.should_commit has already synchronized all gradient work.
            self.work_tracker.clear()
            update()
            import torch_npu

            torch_npu.npu.current_stream().synchronize()
            self.optimizer_update_completed = True
            self._warmup_pending = False