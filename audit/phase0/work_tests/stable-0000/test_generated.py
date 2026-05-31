import pytest
from playwright.sync_api import Page, expect

def test_verify_page_title_contains_playwright(page: Page) -> None:
    page.goto('https://playwright.dev/')
    expect(page.get_by_role("heading", name="Playwright enables reliable web automation for testing, scripting, and AI agents.")).to_have_text('Playwright')