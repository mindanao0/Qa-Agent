import pytest
from playwright.sync_api import Page, expect

def test_dropdown_selection(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/dropdown')
    
    # Step 1: Select Option 2 from the dropdown
    page.get_by_role('combobox').select_option(label='Option 2')
    
    # Step 2: Verify Option 2 is selected
    selected_option = page.get_by_role('combobox').get_attribute('value')
    expect(selected_option).to_be('2')