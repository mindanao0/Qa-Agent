import pytest
from playwright.sync_api import Page, expect

def test_python_language_docs_heading(page: Page) -> None:
    page.goto('https://playwright.dev/')
    heading = page.get_by_text('Python', exact=True)
    expect(heading).to_have_text('Python')