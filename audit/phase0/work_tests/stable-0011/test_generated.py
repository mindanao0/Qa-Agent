import pytest
from playwright.sync_api import Page, expect

def test_todo_crud_operations(page: Page) -> None:
    page.goto('https://demo.playwright.dev/todomvc')
    
    # Step 1: Add a new todo item
    page.get_by_role('textbox').fill('New Todo')
    page.keyboard.press('Enter')
    expect(page.get_by_text('New Todo')).to_be_visible()
    
    # Step 2: Double-click the added todo to edit it
    page.get_by_text('New Todo').dblclick()
    expect(page.get_by_role('textbox', name='What needs to be done?')).to_be_visible()
    
    # Step 3: Change the text of the todo
    page.get_by_role('textbox', name='What needs to be done?').fill('Updated Todo')
    page.keyboard.press('Enter')
    expect(page.get_by_text('Updated Todo')).to_be_visible()
    
    # Step 4: Delete the updated todo
    page.get_by_text('Updated Todo').hover()
    page.get_by_text('real TodoMVC app.').click()
    expect(page.get_by_text('Updated Todo')).not_to_be_visible()