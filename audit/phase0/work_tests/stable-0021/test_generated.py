import pytest
from playwright.sync_api import Page, expect

def test_authentication_invalid_username(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/login')
    page.get_by_label('Username').fill('invalid_username')
    page.get_by_label('Password').fill('SuperSecretPassword!')
    page.get_by_role('button', name='Login').click()
    page.get_by_role("heading", name="todos").should_be_visible()