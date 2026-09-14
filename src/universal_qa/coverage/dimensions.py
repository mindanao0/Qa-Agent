"""Coverage cross-check: merge independent "how much of the site did we see"
signals into one report.

Each dimension answers the same question via a different method with a
different blind spot (declared-vs-observed, DOM-crawl-vs-executed-code, ...).
No single dimension — human-curated golden dataset included — is treated as
ground truth here; agreement across dimensions is the signal, and a path
seen by only one method is a lead to chase, not proof of a gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DimensionResult:
    method: str  # e.g. "sfg_crawl", "static_declared", "js_css_coverage"
    target_id: str
    paths: set[str] = field(default_factory=set)
    meta: dict = field(default_factory=dict)


def merge_dimensions(results: list[DimensionResult]) -> dict:
    """Cross-check dimensions that report a path set; report scalar-only
    dimensions (e.g. js_css_coverage, which has no path set) separately.
    """
    path_dims = [r for r in results if r.paths]
    scalar_dims = [r for r in results if not r.paths]

    if not path_dims:
        return {
            "path_dimensions": [],
            "scalar_metrics": {r.method: r.meta for r in scalar_dims},
        }

    all_paths: set[str] = set()
    for r in path_dims:
        all_paths |= r.paths

    per_path_sources = {
        p: sorted(r.method for r in path_dims if p in r.paths)
        for p in sorted(all_paths)
    }
    n_methods = len(path_dims)
    seen_by_all = sorted(p for p, srcs in per_path_sources.items() if len(srcs) == n_methods)

    exclusive_per_method: dict[str, list[str]] = {}
    for r in path_dims:
        others: set[str] = set()
        for o in path_dims:
            if o.method != r.method:
                others |= o.paths
        exclusive_per_method[r.method] = sorted(r.paths - others)

    return {
        "methods": [r.method for r in path_dims],
        "total_distinct_paths": len(all_paths),
        "seen_by_all_methods": seen_by_all,
        "seen_by_all_count": len(seen_by_all),
        "exclusive_per_method": exclusive_per_method,
        "per_path_sources": per_path_sources,
        "scalar_metrics": {r.method: r.meta for r in scalar_dims},
    }
