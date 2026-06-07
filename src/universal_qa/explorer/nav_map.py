from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ElementCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    role: str | None = None
    name: str | None = None
    selector: str | None = None
    is_in_iframe: bool = False
    is_in_shadow: bool = False
    priority: int = 2  # 0=nav, 1=form-button, 2=other


class ExploredAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_url: str
    action_label: str
    element_role: str | None = None
    element_name: str | None = None
    element_selector: str | None = None
    leads_to_url: str | None = None
    leads_to_modal: bool = False
    state_change: dict | None = None
    is_destructive: bool = False


class ExploredPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    pam_content: str
    actions: list[ExploredAction]
    state_snapshot: str | None = None
    requires_path: list[ExploredAction] = Field(default_factory=list)


class NavigationFlow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow_id: str
    name: str
    steps: list[ExploredAction]
    start_url: str
    end_url: str


class NavigationMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    pages: list[ExploredPage]
    flows: list[NavigationFlow]
    explored_at_iso: str


class ExplorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_pages: int = 50
    explore_timeout_min: int = 5
    max_depth: int = 4
    allow_destructive: bool = False
    max_visits_per_url: int = 2


__all__ = [
    "ElementCandidate", "ExploredAction", "ExploredPage",
    "NavigationFlow", "NavigationMap", "ExplorerConfig",
]
