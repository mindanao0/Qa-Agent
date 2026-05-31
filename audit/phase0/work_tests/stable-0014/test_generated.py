import pytest
from playwright.sync_api import Page, expect

def test_add_and_complete_todos(page: Page) -> None:
    page.goto('https://demo.playwright.dev/todomvc')
    
    # Step 1: Add a new todo item
    page.get_by_role('textbox', name='What needs to be done?').fill('Buy groceries')
    page.get_by_text('real TodoMVC app.').click()
    
    # Expected result: The input field should be cleared after adding the todo.
    expect(page.get_by_role('textbox', name='What needs to be done?')).to_have_value('')
    
    # Step 2: Mark all todos complete using the toggle-all checkbox
    page.get_by_role('checkbox', name='Toggle All').click()
    
    # Expected result: All todos should be marked as completed and the list should refresh to reflect this change.
    expect(page.get_by_text('Buy groceries')).to_have_class('completed')