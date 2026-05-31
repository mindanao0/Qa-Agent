import pytest
from playwright.sync_api import Page, expect

def test_dynamic_content_link_verification(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/dynamic_content')
    dynamic_content_link = page.get_by_role('link', name='Dynamic Content')
    dynamic_content_link.click()
    expect(page.locator('h1')).to_have_text('Dynamic Content')