import pytest
from playwright.sync_api import Page, expect

def test_verify_checkboxes_visibility(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/checkboxes')
    expect(page).to_have_url('https://the-internet.herokuapp.com/checkboxes')

    checkbox1 = page.get_by_role('checkbox', name='Checkbox 1')
    expect(checkbox1).to_be_visible()

    checkbox2 = page.get_by_role('checkbox', name='Checkbox 2')
    expect(checkbox2).to_be_visible()