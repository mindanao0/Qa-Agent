import pytest
from playwright.sync_api import Page, expect

def test_verify_docs_page_loads_after_get_started_link(page: Page) -> None:
    page.goto('https://playwright.dev/')
    page.get_by_text('Get started').click()
    current_url = page.url
    expect(current_url).to_have_url('https://playwright.dev/docs')
    expect(page.get_by_text('Welcome to Playwright')).to_be_visible()