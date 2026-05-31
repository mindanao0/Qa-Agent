import pytest
from playwright.sync_api import Page, expect

def test_add_two_todo_items_and_verify_count(page: Page) -> None:
    page.goto('https://demo.playwright.dev/todomvc')
    
    # Add the first todo item
    page.get_by_placeholder('What needs to be done?').fill('Buy groceries')
    page.keyboard.press('Enter')
    
    # Add the second todo item
    page.get_by_placeholder('What needs to be done?').fill('Clean the house')
    page.keyboard.press('Enter')
    
    # Verify that the count shows 2 items left
    expect(page.get_by_role("heading", name="todos")).to_have_count(2)