import pytest
from playwright.sync_api import Page, expect

def test_todo_list_crud_operations(page: Page) -> None:
    page.goto('https://demo.playwright.dev/todomvc')
    
    # Step 2: Add a new todo item.
    page.get_by_placeholder('What needs to be done?').fill('Task 1')
    page.get_by_role('button', name='Add').click()
    
    # Step 3: Repeat step 2 to add four more todos.
    page.get_by_placeholder('What needs to be done?').fill('Task 2')
    page.get_by_role('button', name='Add').click()
    page.get_by_placeholder('What needs to be done?').fill('Task 3')
    page.get_by_role('button', name='Add').click()
    page.get_by_placeholder('What needs to be done?').fill('Task 4')
    page.get_by_role('button', name='Add').click()
    page.get_by_placeholder('What needs to be done?').fill('Task 5')
    page.get_by_role('button', name='Add').click()
    
    # Step 4: Verify that all todos are initially marked as incomplete.
    expect(page.get_by_role('listitem').filter(has_text='Task 1')).to_be_visible()
    expect(page.get_by_role('listitem').filter(has_text='Task 2')).to_be_visible()
    expect(page.get_by_role('listitem').filter(has_text='Task 3')).to_be_visible()
    expect(page.get_by_role('listitem').filter(has_text='Task 4')).to_be_visible()
    expect(page.get_by_role('listitem').filter(has_text='Task 5')).to_be_visible()
    
    # Step 5: Click the Active filter to display only incomplete todos.
    page.get_by_role('button', name='Active').click()
    
    # Verify that only the five incomplete todos are displayed
    expect(page.get_by_role('listitem').filter(has_text='Task 1')).to_be_visible()
    expect(page.get_by_role('listitem').filter(has_text='Task 2')).to_be_visible()
    expect(page.get_by_role('listitem').filter(has_text='Task 3')).to_be_visible()
    expect(page.get_by_role('listitem').filter(has_text='Task 4')).to_be_visible()
    expect(page.get_by_role('listitem').filter(has_text='Task 5')).to_be_visible()