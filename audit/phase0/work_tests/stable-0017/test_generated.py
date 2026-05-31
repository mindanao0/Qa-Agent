import pytest
from playwright.sync_api import Page, expect

def test_dropdown_validation(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/')
    page.get_by_text('Dropdown').click()
    expect(page.get_by_text('Please select an option')).to_be_visible()