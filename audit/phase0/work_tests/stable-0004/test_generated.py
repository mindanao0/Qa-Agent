import pytest
from playwright.sync_api import Page, expect

def test_verify_docs_link_navigation(page: Page) -> None:
    page.goto('https://playwright.dev/')
    docs_link = page.get_by_role('navigation').get_by_text('Docs')
    expect(docs_link).to_be_visible()
    docs_link.click()
    expect(page).to_have_url('https://playwright.dev/')