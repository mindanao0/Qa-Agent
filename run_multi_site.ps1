$sites = @(
    # E-commerce with login — cart, checkout, order flows
    @{ label="practicesoftwaretesting"; url="https://practicesoftwaretesting.com"; username="customer@practicesoftwaretesting.com"; password="welcome01" },
    # Misc UI demos — alerts, frames, drag-drop, file upload, dynamic content
    @{ label="the-internet"; url="https://the-internet.herokuapp.com"; username=""; password="" },
    # Widget-heavy — date picker, sliders, accordions, modals
    @{ label="demoqa"; url="https://demoqa.com"; username=""; password="" },
    # Large e-commerce (Magento) — search, product detail, categories
    @{ label="magento"; url="https://magento.softwaretestingboard.com"; username=""; password="" },
    # Full e-commerce — register/login, wishlist, checkout, contact
    @{ label="automationexercise"; url="https://automationexercise.com"; username=""; password="" },
    # HR system — dashboard, employee, leave, recruitment (login: Admin/admin123)
    @{ label="orangehrm"; url="https://opensource-demo.orangehrmlive.com"; username="Admin"; password="admin123" },
    # Banking — accounts, transfers, loan (login: john/demo)
    @{ label="parabank"; url="https://parabank.parasoft.com"; username="john"; password="demo" },
    # nopCommerce — full shop with register/login/checkout
    @{ label="nopcommerce"; url="https://demo.nopcommerce.com"; username=""; password="" }
)

foreach ($site in $sites) {
    Write-Host "========================================"
    Write-Host "SITE: $($site.label)"
    Write-Host "URL:  $($site.url)"
    Write-Host "========================================"
    $out = "reports/multi_site/$($site.label).txt"
    if ($site.username) {
        uv run python -m src.universal_qa --url $site.url --username $site.username --password $site.password --max-pages 30 2>&1 | Tee-Object -FilePath $out
    } else {
        uv run python -m src.universal_qa --url $site.url --max-pages 30 2>&1 | Tee-Object -FilePath $out
    }
    Write-Host ""
}

Write-Host "ALL DONE"
