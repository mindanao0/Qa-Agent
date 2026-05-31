import pytest
from playwright.sync_api import Page, expect

def test_github_link_in_footer(page: Page) -> None:
    page.goto('https://playwright.dev/')
    
    # Step 2: Verify that the GitHub repository link is present in the footer.
    github_link = page.get_by_text('Fork me on GitHub')
    expect(github_link).to_be_visible()
    
    # Step 3: Verify that the footer is displayed correctly after navigating to the GitHub repository link.
    github_link.click()
    expect(page).to_have_url('https://github.com/microsoft/playwright')
    expect(page.get_by_text('Fork me on GitHub')).to_be_visible()