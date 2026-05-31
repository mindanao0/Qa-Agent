import pytest
from playwright.sync_api import Page, expect

def test_checkbox_selection(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/checkboxes')
    checkbox = page.get_by_role('checkbox')
    checkbox.check()
    expect(checkbox).to_be_checked()