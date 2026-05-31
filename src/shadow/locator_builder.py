# src/shadow/locator_builder.py
from __future__ import annotations


def build(host_selector: str, inner_selector: str) -> str:
    """Return a Playwright CSS pierce locator: `host >> css=inner`.

    Raises ValueError if inner_selector is an absolute XPath (starts with /).
    """
    if inner_selector.startswith("/"):
        raise ValueError(
            f"inner_selector must not be an absolute XPath: {inner_selector!r}"
        )
    return f"{host_selector} >> css={inner_selector}"


def build_chain(selectors: list[str]) -> str:
    """Multi-level pierce: ``host >> css=slot >> css=button``.

    Raises ValueError if fewer than 2 selectors are supplied or any selector
    after the first starts with / (absolute XPath).
    """
    if len(selectors) < 2:
        raise ValueError("build_chain requires at least 2 selectors")
    result = selectors[0]
    for sel in selectors[1:]:
        if sel.startswith("/"):
            raise ValueError(
                f"Selector must not be an absolute XPath: {sel!r}"
            )
        result = f"{result} >> css={sel}"
    return result


__all__ = ["build", "build_chain"]
