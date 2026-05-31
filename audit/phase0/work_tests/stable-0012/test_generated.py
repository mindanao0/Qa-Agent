import pytest
from playwright.sync_api import Page, expect

def test_verify_input_placeholder_text(page: Page) -> None:
    page.goto('https://demo.playwright.dev/todomvc')
    input_field = page.get_by_placeholder('What needs to be done')
    expect(input_field).to_be_visible()
    assert input_field.get_attribute('placeholder') == 'What needs to be done'