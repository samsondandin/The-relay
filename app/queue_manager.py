"""Queue manager and admission controller for concurrent request scheduling."""

import time
import asyncio
from typing import Optional
from contextlib import asynccontextmanager


class QueueFullException(Exception):
    """Raised when incoming requests exceed the maximum allowed admission queue capacity."""
    def __init__(self, message: str = "Admission queue is full. Request rejected under heavy load."):
        super().__init__(message)


class QueueTimeoutException(Exception):
    """Raised when a request waits in the queue longer than the allowed timeout."""
    def __init__(self, message: str = "Request timed out waiting in admission queue."):
        super().__init__(message)


class AdmissionQueueManager:
    """Bounded FIFO admission controller regulating concurrency and measuring queue wait times."""

    def __init__(
        self,
        max_concurrency: int = 5,
        queue_capacity: int = 20,
        queue_timeout_seconds: float = 10.0,
    ):
        self.max_concurrency = max_concurrency
        self.queue_capacity = queue_capacity
        self.queue_timeout_seconds = queue_timeout_seconds

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._current_waiting = 0
        self._active_concurrency = 0
        self._lock = asyncio.Lock()

    @property
    def current_waiting(self) -> int:
        return self._current_waiting

    @property
    def active_concurrency(self) -> int:
        return self._active_concurrency

    @asynccontextmanager
    async def acquire_slot(self):
        """Context manager that enqueues the request, enforces queue bounds,

        measures queue wait time, acquires an execution slot, and releases upon completion.
        """
        async with self._lock:
            if self._current_waiting >= self.queue_capacity:
                raise QueueFullException(
                    f"Queue capacity of {self.queue_capacity} exceeded (currently waiting: {self._current_waiting})"
                )
            self._current_waiting += 1

        t_start = time.perf_counter()
        acquired = False
        try:
            # Wait for available concurrency slot up to queue_timeout_seconds
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.queue_timeout_seconds
            )
            acquired = True
            t_acquired = time.perf_counter()
            wait_ms = (t_acquired - t_start) * 1000.0

            async with self._lock:
                self._current_waiting = max(0, self._current_waiting - 1)
                self._active_concurrency += 1

            yield wait_ms

        except asyncio.TimeoutError:
            async with self._lock:
                self._current_waiting = max(0, self._current_waiting - 1)
            raise QueueTimeoutException(
                f"Timed out after {self.queue_timeout_seconds}s waiting in queue"
            )
        finally:
            if acquired:
                async with self._lock:
                    self._active_concurrency = max(0, self._active_concurrency - 1)
                self._semaphore.release()

