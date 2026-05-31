import pytest
from playwright.sync_api import Page, expect

def test_nodejs_dropdown_validation(page: Page) -> None:
    page.goto('https://playwright.dev/')
    page.get_by_role('button', name='Node.js').click()
    expect(page.get_by_role("link", name="Python")).to_be_visible()