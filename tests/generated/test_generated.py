import pytest
from playwright.sync_api import Page, expect

def test_login_with_valid_credentials(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/login')
    page.get_by_label('Username').fill('admin')
    page.get_by_label('Password').fill('admin')
    page.get_by_role('button', name='Login').click()
    expect(page).to_have_url('https://the-internet.herokuapp.com/dashboard')

def test_login_with_invalid_credentials(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/login')
    page.get_by_label('Username').fill('nonexistent')
    page.get_by_label('Password').fill('nonexistent')
    page.get_by_role('button', name='Login').click()
    expect(page.get_by_text('Your username is invalid!')).to_be_visible()

def test_login_with_empty_credentials(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/login')
    page.get_by_label('Username').fill('')
    page.get_by_label('Password').fill('')
    page.get_by_role('button', name='Login').click()
    expect(page.get_by_text('Your username is invalid!')).to_be_visible()

def test_login_with_invalid_password(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/login')
    page.get_by_label('Username').fill('admin')
    page.get_by_label('Password').fill('invalid')
    page.get_by_role('button', name='Login').click()
    expect(page.get_by_role('heading', name='This is where you can log into the secure area. Enter tomsmith for the username and SuperSecretPassword! for the password. If the information is wrong you should see error messages.')).to_be_visible()