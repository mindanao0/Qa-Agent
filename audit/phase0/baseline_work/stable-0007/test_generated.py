import pytest
from playwright.sync_api import Page, expect

def test_add_new_todo_item_and_verify_appearance_in_list(page: Page) -> None:
    page.goto('https://demo.playwright.dev/todomvc')
    
    # Add a new todo item titled 'Buy milk'
    page.get_by_placeholder('What needs to be done?').fill('Buy milk')
    page.keyboard.press('Enter')

    # Verify that the list refreshes after adding a new item
    expect(page.get_by_role('listitem').nth(0).get_by_text('Buy milk')).to_be_visible()