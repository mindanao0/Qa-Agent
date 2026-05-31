"""Priority 5 Observer-Driver multi-agent system.

The Driver agent performs state-mutating browser actions; the Observers
consume the same trace stream asynchronously to run audits without
contending for browser state.
"""
from src.agents.observer_driver.coordinator import build_graph
from src.agents.observer_driver.state import (
    AgentState,
    DriverAction,
    ObserverReport,
)
from src.agents.observer_driver.trace_bus import TraceBus, TraceEvent

__all__ = [
    "AgentState",
    "DriverAction",
    "ObserverReport",
    "TraceBus",
    "TraceEvent",
    "build_graph",
]
