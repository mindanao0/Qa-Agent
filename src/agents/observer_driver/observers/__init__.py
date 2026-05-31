"""Observer agents — passive auditors of the Driver's trace stream.

Each observer subscribes to a TraceBus queue, processes events independently,
and emits :class:`~src.agents.observer_driver.state.ObserverReport` items.
Observers never mutate browser state.
"""
from src.agents.observer_driver.observers.accessibility_observer import (
    AccessibilityObserver,
)
from src.agents.observer_driver.observers.base_observer import BaseObserver
from src.agents.observer_driver.observers.performance_observer import (
    PerformanceObserver,
)
from src.agents.observer_driver.observers.security_observer import SecurityObserver

__all__ = [
    "AccessibilityObserver",
    "BaseObserver",
    "PerformanceObserver",
    "SecurityObserver",
]
