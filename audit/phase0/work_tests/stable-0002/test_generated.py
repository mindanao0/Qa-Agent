import pytest
from playwright.sync_api import Page, expect

def test_search_button_visibility(page: Page) -> None:
    page.goto('https://playwright.dev/')
    search_button = page.get_by_role('button', name='Search')
    expect(search_button).to_be_visible()