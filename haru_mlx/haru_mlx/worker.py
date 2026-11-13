"""The single thread that runs all MLX work."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable


class MLXWorker:
    """Runs submitted calls one after another on one thread.

    MLX keeps its GPU streams per thread, so the model is loaded and every
    conversation is driven on this thread.

    Between frames the thread waits. From ``warm`` seconds before the next
    frame is due (see :meth:`expect`) it polls instead of blocking: a CPU that
    idled through the gap between two frames starts the next one a few
    milliseconds slower, and a ``sleep(0)`` loop keeps it awake while still
    releasing the interpreter lock to the event loop.
    """

    LATE = 0.1  # stop polling when a frame is this late (the input paused)

    def __init__(self, warm: float = 0.03):
        self.warm = warm
        self._due: float | None = None
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="haru-mlx", daemon=True).start()

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        """Queue ``fn(*args)``; the future resolves on the worker thread."""
        future: Future = Future()
        self._jobs.put((fn, args, future))
        return future

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Call ``fn(*args)`` on the worker thread from a coroutine."""
        return await asyncio.wrap_future(self.submit(fn, *args))

    def expect(self, due: float | None) -> None:
        """``time.perf_counter()`` at which the next frame should arrive, or None when idle."""
        self._due = due

    def _next(self):
        while True:
            due = self._due
            if due is None or self.warm <= 0:
                return self._jobs.get()
            now = time.perf_counter()
            if now < due - self.warm:
                try:
                    return self._jobs.get(timeout=due - self.warm - now)
                except queue.Empty:
                    continue
            if now > due + self.LATE:
                self._due = None
                continue
            try:
                return self._jobs.get_nowait()
            except queue.Empty:
                time.sleep(0)

    def _run(self) -> None:
        while True:
            fn, args, future = self._next()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as error:  # delivered to whoever waits on the future
                future.set_exception(error)
