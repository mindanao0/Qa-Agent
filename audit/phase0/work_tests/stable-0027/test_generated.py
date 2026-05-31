import pytest
from playwright.sync_api import Page, expect

def test_form_input_validation(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/inputs')
    page.get_by_role('spinbutton').fill('42')
    expect(page.get_by_role('spinbutton')).to_have_value('42')