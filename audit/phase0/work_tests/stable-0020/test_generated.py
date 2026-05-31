import pytest
from playwright.sync_api import Page, expect

def test_successful_login(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/login')
    username_input = page.get_by_label("Username")
    password_input = page.get_by_label("Password")
    login_button = page.get_by_text("Login")

    username_input.fill('tomsmith')
    password_input.fill('SuperSecretPassword!')
    login_button.click()

    expect(page).to_have_url('https://the-internet.herokuapp.com/secure')

def test_failed_login(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/login')
    username_input = page.get_by_label("Username")
    password_input = page.get_by_label("Password")
    login_button = page.get_by_text("Login")

    username_input.fill('tomsmith')
    password_input.fill('wrongpassword!')
    login_button.click()

    expect(page).to_have_url('https://the-internet.herokuapp.com/login')
    error_message = page.get_by_text("Your credentials provided do not allow you to log in.")
    expect(error_message).to_be_visible()