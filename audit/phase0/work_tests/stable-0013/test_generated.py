import pytest
from playwright.sync_api import Page, expect

def test_todo_list_crud_operations(page: Page) -> None:
    page.goto('https://demo.playwright.dev/todomvc')

    # Add the first todo item
    page.get_by_placeholder('What needs to be done?').fill('Task 1')
    page.keyboard.press('Enter')
    expect(page.get_by_text('Task 1')).to_be_visible()

    # Add the second todo item
    page.get_by_placeholder('What needs to be done?').fill('Task 2')
    page.keyboard.press('Enter')
    expect(page.get_by_text('Task 2')).to_be_visible()

    # Add the third todo item
    page.get_by_placeholder('What needs to be done?').fill('Task 3')
    page.keyboard.press('Enter')
    expect(page.get_by_text('Task 3')).to_be_visible()

    # Add the fourth todo item
    page.get_by_placeholder('What needs to be done?').fill('Task 4')
    page.keyboard.press('Enter')
    expect(page.get_by_text('Task 4')).to_be_visible()

    # Add the fifth todo item
    page.get_by_placeholder('What needs to be done?').fill('Task 5')
    page.keyboard.press('Enter')
    expect(page.get_by_text('Task 5')).to_be_visible()

    # Click on the 'Active' filter button
    page.get_by_text('Active').click()
    expect(page.get_by_text('Task 1')).to_be_visible()
    expect(page.get_by_text('Task 2')).to_be_visible()
    expect(page.get_by_text('Task 3')).to_be_visible()
    expect(page.get_by_text('Task 4')).to_be_visible()
    expect(page.get_by_text('Task 5')).to_be_visible()