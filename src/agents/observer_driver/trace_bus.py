"""asyncio.Queue-based fan-out event bus connecting Driver → Observers.

The Driver publishes :class:`TraceEvent` items as it executes; every Observer
holds its own :class:`asyncio.Queue` subscription. Publication is strictly
non-blocking (``put_nowait``) so a slow or stuck Observer cannot back-pressure
the Driver — full queues drop the oldest event and log a warning.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

QUEUE_MAX_SIZE = 100


@dataclass
class TraceEvent:
    """Single event broadcast on the bus."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int = 0


class TraceBus:
    """Many-to-many publish/subscribe over bounded asyncio Queues.

    *publish()* fans an event out to every subscriber queue without ever
    awaiting — if a queue is full, the oldest item is dropped to make room.
    This guarantees the Driver coroutine is never blocked by an Observer.
    """

    def __init__(self, queue_max_size: int = QUEUE_MAX_SIZE) -> None:
        self._queue_max_size = queue_max_size
        self.subscribers: list[asyncio.Queue[TraceEvent]] = []

    def subscribe(self) -> asyncio.Queue[TraceEvent]:
        """Register a new subscriber and return its private queue."""
        queue: asyncio.Queue[TraceEvent] = asyncio.Queue(maxsize=self._queue_max_size)
        self.subscribers.append(queue)
        return queue

    async def publish(self, event: TraceEvent) -> None:
        """Broadcast *event* to every subscriber queue, never blocking.

        The function is declared ``async`` for forward compatibility but
        contains no ``await`` — it returns synchronously after the fan-out.
        """
        for queue in self.subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest event to make room for the new one rather
                # than blocking. This is the explicit "Driver wins" policy.
                try:
                    _ = queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        "TraceBus: dropped event after eviction "
                        f"(event_type={event.event_type!r})"
                    )

    def close(self) -> None:
        """Detach all subscribers (events already queued remain consumable)."""
        self.subscribers.clear()


__all__ = ["TraceBus", "TraceEvent", "QUEUE_MAX_SIZE"]
