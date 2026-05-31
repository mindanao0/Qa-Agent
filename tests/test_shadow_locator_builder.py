# tests/test_shadow_locator_builder.py
from __future__ import annotations

import pytest

from src.shadow.locator_builder import build, build_chain


def test_build_simple():
    assert build("my-card", "button.submit") == "my-card >> css=button.submit"


def test_build_with_compound_inner():
    assert build("custom-input", "input[type=text]") == "custom-input >> css=input[type=text]"


def test_build_rejects_absolute_xpath():
    with pytest.raises(ValueError, match="absolute XPath"):
        build("host", "/html/body/button")


def test_build_rejects_rooted_xpath_with_dot():
    # "/button" starts with "/" — also an absolute XPath
    with pytest.raises(ValueError, match="absolute XPath"):
        build("host", "/button")


def test_build_chain_two_levels():
    result = build_chain(["my-host", "div.slot", "button"])
    assert result == "my-host >> css=div.slot >> css=button"


def test_build_chain_three_levels():
    result = build_chain(["outer-elem", "inner-elem", "span", "a"])
    assert result == "outer-elem >> css=inner-elem >> css=span >> css=a"


def test_build_chain_requires_at_least_two():
    with pytest.raises(ValueError, match="at least 2"):
        build_chain(["only-one"])


def test_build_chain_rejects_xpath_in_any_position():
    with pytest.raises(ValueError, match="absolute XPath"):
        build_chain(["host", "/body/button", "span"])


def test_build_chain_empty_raises():
    with pytest.raises(ValueError, match="at least 2"):
        build_chain([])
