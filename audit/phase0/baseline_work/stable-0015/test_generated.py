import pytest
from playwright.sync_api import Page, expect

def test_authentication_page_verification(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/')
    page.get_by_text('Form Authentication').click()
    expect(page).to_have_url('https://the-internet.herokuapp.com/login')