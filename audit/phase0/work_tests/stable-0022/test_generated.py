import pytest
from playwright.sync_api import Page, expect

def test_login_with_blank_fields(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/login')
    
    # Step 2: Leave both the username and password fields blank.
    page.fill('input[name="username"]', '')
    page.fill('input[name="password"]', '')

    # Step 3: Click the Login button.
    page.click('button:has-text("Login")')

    # Step 4: Verify that an error message appears for the username field.
    expect(page.get_by_role('textbox', name='Username')).to_be_visible()

    # Step 5: Verify that an error message appears for the password field.
    expect(page.get_by_role('textbox', name='Username')).to_be_visible()