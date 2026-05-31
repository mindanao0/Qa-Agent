import pytest
from playwright.sync_api import Page, expect

def test_verify_user_images_visibility_on_hovers_page(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/hovers')
    user_images = page.get_by_role('img')
    expect(user_images).to_have_count(3)
    for image in user_images.all():
        expect(image).to_be_visible()