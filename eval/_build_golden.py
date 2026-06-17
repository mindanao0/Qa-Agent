"""Builder for eval/golden_set.jsonl — the test-planner golden benchmark.

Provenance / honesty note
--------------------------
The 20 pages below are REAL, publicly reachable QA demo targets (saucedemo,
the-internet.herokuapp.com, parabank, demoqa, orangehrm, automationexercise)
plus the Conduit/RealWorld app at realworld.habsida.net that prior sprints
already drove. Element lists reflect each site's *canonical, stable* structure
and were cross-checked against the real explored elements mined from
data/training/train.jsonl (universal_qa_* sources). They are a fixed, curated
spec — NOT a live DOM scrape — which is exactly what we want for an eval: the
SAME input feeds every planner config (baseline / +grammar / +few-shot / +RAG),
so any imperfection cancels out in the deltas.

Each page declares:
  - url, title, pam_content, requires_auth
  - elements: interactive elements the planner should produce tests for
              (role/name → ExploredAction; is_form marks inputs/submit buttons)
  - expected_test_count: principled count of meaningful test scenarios (for the
              secondary count-based coverage diagnostic)
  - expected_assertion_types: concrete state/backend assertions the page warrants
              (vocabulary: redirect, validation_error, content_match, count_change,
               state_persist, http_status, wcag) — never "toBeVisible"
  - expected_scenarios: the behaviours that SHOULD be covered. Each has bilingual
              (TH/EN) keywords_any; a scenario counts as covered if any keyword
              appears in any generated test for the page. This drives the PRIMARY,
              non-saturating coverage_score.

Run: python eval/_build_golden.py   ->   writes eval/golden_set.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path


def el(role: str, name: str, is_form: bool = False, leads_to_url: str | None = None) -> dict:
    return {"role": role, "name": name, "is_form": is_form, "leads_to_url": leads_to_url}


def sc(sid: str, *keywords: str) -> dict:
    return {"id": sid, "keywords_any": list(keywords)}


PAGES: list[dict] = [
    # ───────────────────────── saucedemo ─────────────────────────
    {
        "page_id": "saucedemo_login", "site": "saucedemo",
        "url": "https://www.saucedemo.com/", "title": "Swag Labs — Login",
        "pam_content": "Login form: Username and Password textboxes and a Login button. "
                       "An error banner appears for invalid credentials. standard_user/secret_sauce "
                       "succeeds; locked_out_user is blocked with an error.",
        "requires_auth": False,
        "elements": [
            el("textbox", "Username", True),
            el("textbox", "Password", True),
            el("button", "Login", True, "https://www.saucedemo.com/inventory.html"),
        ],
        "expected_test_count": 6,
        "expected_assertion_types": ["redirect", "validation_error", "content_match"],
        "expected_scenarios": [
            sc("valid_login", "inventory", "เข้าสู่ระบบ", "สำเร็จ", "หน้าสินค้า", "products"),
            sc("locked_out", "locked", "ล็อก", "ถูกล็อก", "locked_out"),
            sc("wrong_password", "ไม่ถูกต้อง", "error", "ผิดพลาด", "ไม่ถูก", "invalid"),
            sc("empty_username", "required", "ต้องกรอก", "ว่าง", "username is required", "ชื่อผู้ใช้"),
            sc("empty_password", "password is required", "รหัสผ่าน", "ต้องกรอกรหัส"),
        ],
    },
    {
        "page_id": "saucedemo_inventory", "site": "saucedemo",
        "url": "https://www.saucedemo.com/inventory.html", "title": "Swag Labs — Products",
        "pam_content": "Product listing after login. Each product has an Add to cart button that "
                       "toggles to Remove and increments the cart badge. A sort dropdown reorders "
                       "products. The shopping cart link opens the cart page.",
        "requires_auth": True,
        "elements": [
            el("button", "Add to cart", True),
            el("combobox", "Sort", True),
            el("link", "Shopping cart", False, "https://www.saucedemo.com/cart.html"),
            el("button", "Open Menu", False),
            el("link", "Sauce Labs Backpack", False, "https://www.saucedemo.com/inventory-item.html?id=4"),
        ],
        "expected_test_count": 7,
        "expected_assertion_types": ["count_change", "redirect", "state_persist", "content_match"],
        "expected_scenarios": [
            sc("add_increments_badge", "ตะกร้า", "badge", "จำนวน", "เพิ่มขึ้น", "cart", "count", "1"),
            sc("sort_reorders", "เรียง", "sort", "ลำดับ", "ราคา", "a to z", "z to a", "order"),
            sc("open_cart", "cart.html", "ตะกร้า", "หน้าตะกร้า", "cart page"),
            sc("product_detail", "inventory-item", "รายละเอียดสินค้า", "detail", "สินค้า"),
            sc("remove_decrements", "remove", "นำออก", "ลดลง", "เอาออก"),
        ],
    },
    {
        "page_id": "saucedemo_cart", "site": "saucedemo",
        "url": "https://www.saucedemo.com/cart.html", "title": "Swag Labs — Cart",
        "pam_content": "Cart page listing chosen items. Checkout button proceeds to checkout step one. "
                       "Continue Shopping returns to inventory. Remove deletes an item and decrements the badge.",
        "requires_auth": True,
        "elements": [
            el("button", "Checkout", True, "https://www.saucedemo.com/checkout-step-one.html"),
            el("button", "Continue Shopping", False, "https://www.saucedemo.com/inventory.html"),
            el("button", "Remove", False),
        ],
        "expected_test_count": 5,
        "expected_assertion_types": ["redirect", "count_change", "state_persist"],
        "expected_scenarios": [
            sc("checkout_proceeds", "checkout-step-one", "ชำระเงิน", "checkout", "ข้อมูลผู้ซื้อ"),
            sc("continue_back", "inventory", "เลือกซื้อต่อ", "กลับ", "หน้าสินค้า"),
            sc("remove_item", "remove", "นำออก", "ลบ", "ลดลง", "badge", "ตะกร้า"),
        ],
    },
    {
        "page_id": "saucedemo_checkout_info", "site": "saucedemo",
        "url": "https://www.saucedemo.com/checkout-step-one.html", "title": "Swag Labs — Checkout: Your Information",
        "pam_content": "Checkout step one form: First Name, Last Name, Zip/Postal Code textboxes; "
                       "Continue proceeds to the overview; Cancel returns to the cart. Missing fields show errors.",
        "requires_auth": True,
        "elements": [
            el("textbox", "First Name", True),
            el("textbox", "Last Name", True),
            el("textbox", "Zip/Postal Code", True),
            el("button", "Continue", True, "https://www.saucedemo.com/checkout-step-two.html"),
            el("button", "Cancel", False, "https://www.saucedemo.com/cart.html"),
        ],
        "expected_test_count": 6,
        "expected_assertion_types": ["validation_error", "redirect"],
        "expected_scenarios": [
            sc("all_filled_proceeds", "checkout-step-two", "ขั้นตอนถัดไป", "overview", "สรุป", "ดำเนินการ"),
            sc("empty_first_name", "first name is required", "ชื่อ", "required", "ต้องกรอก"),
            sc("empty_zip", "postal", "zip", "รหัสไปรษณีย์", "required", "ต้องกรอก"),
            sc("cancel_returns", "cart", "ยกเลิก", "กลับ", "ตะกร้า"),
        ],
    },
    # ───────────────────── the-internet.herokuapp.com ─────────────────────
    {
        "page_id": "theinternet_login", "site": "the-internet",
        "url": "https://the-internet.herokuapp.com/login", "title": "The Internet — Form Authentication",
        "pam_content": "Form authentication page. Username and Password textboxes and a Login button. "
                       "tomsmith/SuperSecretPassword! reaches /secure with a success flash; wrong username "
                       "shows 'Your username is invalid!'; wrong password shows 'Your password is invalid!'.",
        "requires_auth": False,
        "elements": [
            el("textbox", "Username", True),
            el("textbox", "Password", True),
            el("button", "Login", True, "https://the-internet.herokuapp.com/secure"),
        ],
        "expected_test_count": 5,
        "expected_assertion_types": ["redirect", "content_match", "validation_error"],
        "expected_scenarios": [
            sc("valid_login", "secure", "logged into a secure area", "เข้าสู่ระบบ", "สำเร็จ", "พื้นที่ปลอดภัย"),
            sc("invalid_username", "username is invalid", "ชื่อผู้ใช้", "ไม่ถูกต้อง", "invalid"),
            sc("invalid_password", "password is invalid", "รหัสผ่าน", "ไม่ถูกต้อง"),
            sc("logout", "logout", "ออกจากระบบ", "login"),
        ],
    },
    {
        "page_id": "theinternet_dropdown", "site": "the-internet",
        "url": "https://the-internet.herokuapp.com/dropdown", "title": "The Internet — Dropdown List",
        "pam_content": "A single dropdown with Option 1 and Option 2. Selecting an option makes it the "
                       "selected value. Default has no option selected.",
        "requires_auth": False,
        "elements": [
            el("combobox", "Dropdown", True),
        ],
        "expected_test_count": 3,
        "expected_assertion_types": ["state_persist", "content_match"],
        "expected_scenarios": [
            sc("select_option1", "option 1", "ตัวเลือก", "selected", "ถูกเลือก", "ตัวเลือกที่ 1"),
            sc("select_option2", "option 2", "ตัวเลือกที่ 2", "selected", "ถูกเลือก"),
        ],
    },
    {
        "page_id": "theinternet_checkboxes", "site": "the-internet",
        "url": "https://the-internet.herokuapp.com/checkboxes", "title": "The Internet — Checkboxes",
        "pam_content": "Two checkboxes; the first unchecked, the second checked by default. Clicking toggles "
                       "the checked state.",
        "requires_auth": False,
        "elements": [
            el("checkbox", "checkbox 1", True),
            el("checkbox", "checkbox 2", True),
        ],
        "expected_test_count": 3,
        "expected_assertion_types": ["state_persist"],
        "expected_scenarios": [
            sc("check_first", "checked", "ติ๊ก", "เลือก", "ถูกเลือก", "checkbox"),
            sc("uncheck_second", "unchecked", "ยกเลิกการเลือก", "ไม่ถูกเลือก", "เอาเครื่องหมายออก"),
        ],
    },
    {
        "page_id": "theinternet_add_remove", "site": "the-internet",
        "url": "https://the-internet.herokuapp.com/add_remove_elements/", "title": "The Internet — Add/Remove Elements",
        "pam_content": "An Add Element button that appends a Delete button each click. Each Delete button "
                       "removes one element. The number of Delete buttons reflects how many were added.",
        "requires_auth": False,
        "elements": [
            el("button", "Add Element", True),
            el("button", "Delete", False),
        ],
        "expected_test_count": 4,
        "expected_assertion_types": ["count_change", "state_persist"],
        "expected_scenarios": [
            sc("add_creates_delete", "delete", "เพิ่ม", "ปุ่มลบ", "ปรากฏ", "จำนวน", "count"),
            sc("delete_removes", "ลบ", "หาย", "ลดลง", "removed", "นำออก"),
            sc("add_many", "หลาย", "5", "multiple", "จำนวน"),
        ],
    },
    # ───────────────────────────── parabank ─────────────────────────────
    {
        "page_id": "parabank_login", "site": "parabank",
        "url": "https://parabank.parasoft.com/parabank/index.htm", "title": "ParaBank — Login",
        "pam_content": "Customer login: Username and Password textboxes, Log In button. Register and "
                       "'Forgot login info?' links. Valid login opens the Accounts Overview; invalid shows "
                       "'The username and password could not be verified.'",
        "requires_auth": False,
        "elements": [
            el("textbox", "Username", True),
            el("textbox", "Password", True),
            el("button", "Log In", True, "https://parabank.parasoft.com/parabank/overview.htm"),
            el("link", "Register", False, "https://parabank.parasoft.com/parabank/register.htm"),
            el("link", "Forgot login info?", False),
        ],
        "expected_test_count": 5,
        "expected_assertion_types": ["redirect", "validation_error", "content_match"],
        "expected_scenarios": [
            sc("valid_login", "overview", "บัญชี", "accounts", "เข้าสู่ระบบ", "สำเร็จ"),
            sc("invalid_login", "could not be verified", "ไม่ถูกต้อง", "error", "ผิดพลาด"),
            sc("empty_login", "required", "ต้องกรอก", "ว่าง", "please enter"),
            sc("go_register", "register", "สมัคร", "ลงทะเบียน"),
        ],
    },
    {
        "page_id": "parabank_register", "site": "parabank",
        "url": "https://parabank.parasoft.com/parabank/register.htm", "title": "ParaBank — Register",
        "pam_content": "Registration form: First/Last name, address fields, Username, Password and Confirm. "
                       "Register creates the account ('Your account was created successfully'). Mismatched "
                       "passwords and duplicate usernames are rejected with messages.",
        "requires_auth": False,
        "elements": [
            el("textbox", "First Name", True),
            el("textbox", "Last Name", True),
            el("textbox", "Username", True),
            el("textbox", "Password", True),
            el("textbox", "Confirm", True),
            el("button", "Register", True),
        ],
        "expected_test_count": 7,
        "expected_assertion_types": ["content_match", "validation_error", "state_persist"],
        "expected_scenarios": [
            sc("valid_register", "created successfully", "สร้างบัญชี", "สำเร็จ", "ลงทะเบียนสำเร็จ"),
            sc("password_mismatch", "did not match", "รหัสผ่านไม่ตรง", "ไม่ตรงกัน", "passwords"),
            sc("duplicate_username", "already exists", "มีอยู่แล้ว", "ซ้ำ", "username"),
            sc("empty_required", "required", "ต้องกรอก", "ว่าง", "กรุณากรอก"),
        ],
    },
    {
        "page_id": "parabank_transfer", "site": "parabank",
        "url": "https://parabank.parasoft.com/parabank/transfer.htm", "title": "ParaBank — Transfer Funds",
        "pam_content": "Transfer funds form: Amount textbox, From and To account dropdowns, Transfer button. "
                       "A valid transfer shows 'Transfer Complete!' and changes balances. Non-numeric amount is rejected.",
        "requires_auth": True,
        "elements": [
            el("textbox", "Amount", True),
            el("combobox", "From account", True),
            el("combobox", "To account", True),
            el("button", "Transfer", True),
        ],
        "expected_test_count": 5,
        "expected_assertion_types": ["content_match", "count_change", "validation_error"],
        "expected_scenarios": [
            sc("valid_transfer", "transfer complete", "โอนเงินสำเร็จ", "สำเร็จ", "ยอดเงิน", "balance"),
            sc("balance_changes", "ยอดคงเหลือ", "ลดลง", "เพิ่มขึ้น", "balance", "amount"),
            sc("invalid_amount", "ตัวเลข", "ไม่ใช่ตัวเลข", "invalid", "ผิดพลาด", "numeric"),
        ],
    },
    {
        "page_id": "parabank_billpay", "site": "parabank",
        "url": "https://parabank.parasoft.com/parabank/billpay.htm", "title": "ParaBank — Bill Pay",
        "pam_content": "Bill payment form: Payee Name, Address, Account, Verify Account, Amount, From account, "
                       "Send Payment. Valid payment shows 'Bill Payment Complete'. Mismatched account confirmation "
                       "and non-numeric amount are rejected.",
        "requires_auth": True,
        "elements": [
            el("textbox", "Payee Name", True),
            el("textbox", "Account", True),
            el("textbox", "Verify Account", True),
            el("textbox", "Amount", True),
            el("button", "Send Payment", True),
        ],
        "expected_test_count": 6,
        "expected_assertion_types": ["content_match", "validation_error"],
        "expected_scenarios": [
            sc("valid_payment", "bill payment complete", "ชำระบิลสำเร็จ", "สำเร็จ", "payment"),
            sc("account_mismatch", "ไม่ตรงกัน", "verify", "ยืนยันบัญชี", "mismatch"),
            sc("invalid_amount", "ตัวเลข", "ไม่ใช่ตัวเลข", "numeric", "ผิดพลาด"),
            sc("empty_required", "required", "ต้องกรอก", "ว่าง"),
        ],
    },
    # ───────────────────────────── demoqa ─────────────────────────────
    {
        "page_id": "demoqa_textbox", "site": "demoqa",
        "url": "https://demoqa.com/text-box", "title": "DemoQA — Text Box",
        "pam_content": "Text Box form: Full Name, Email, Current Address, Permanent Address, Submit. On submit "
                       "a card echoes the submitted values. An invalid email shows a red border and no output.",
        "requires_auth": False,
        "elements": [
            el("textbox", "Full Name", True),
            el("textbox", "Email", True),
            el("textbox", "Current Address", True),
            el("button", "Submit", True),
        ],
        "expected_test_count": 5,
        "expected_assertion_types": ["content_match", "validation_error"],
        "expected_scenarios": [
            sc("valid_submit", "แสดงค่า", "output", "ค่าที่กรอก", "card", "ข้อมูลที่ส่ง", "submitted"),
            sc("invalid_email", "อีเมล", "email", "ไม่ถูกต้อง", "invalid", "red", "ขอบแดง"),
            sc("empty_submit", "ว่าง", "ไม่แสดงผล", "no output", "required"),
        ],
    },
    {
        "page_id": "demoqa_practice_form", "site": "demoqa",
        "url": "https://demoqa.com/automation-practice-form", "title": "DemoQA — Practice Form",
        "pam_content": "Student registration form: First/Last Name, Email, Gender radio, Mobile (10 digits), "
                       "Submit. Valid submission opens a 'Thanks for submitting the form' modal echoing values. "
                       "Missing required fields are highlighted; a non-10-digit mobile is rejected.",
        "requires_auth": False,
        "elements": [
            el("textbox", "First Name", True),
            el("textbox", "Last Name", True),
            el("textbox", "Mobile Number", True),
            el("button", "Submit", True),
        ],
        "expected_test_count": 6,
        "expected_assertion_types": ["content_match", "validation_error"],
        "expected_scenarios": [
            sc("valid_submit", "thanks for submitting", "ส่งฟอร์มสำเร็จ", "modal", "ขอบคุณ", "สรุปข้อมูล"),
            sc("missing_required", "required", "ต้องกรอก", "highlight", "แดง", "จำเป็น"),
            sc("invalid_mobile", "10", "เบอร์", "mobile", "หลัก", "ตัวเลข", "ไม่ถูกต้อง"),
        ],
    },
    {
        "page_id": "demoqa_webtables", "site": "demoqa",
        "url": "https://demoqa.com/webtables", "title": "DemoQA — Web Tables",
        "pam_content": "A CRUD table. Add opens a form to create a row (row count increases). Search filters rows. "
                       "Edit updates a row's values. Delete removes a row (row count decreases).",
        "requires_auth": False,
        "elements": [
            el("button", "Add", True),
            el("searchbox", "Search", True),
            el("button", "Edit", False),
            el("button", "Delete", False),
        ],
        "expected_test_count": 6,
        "expected_assertion_types": ["count_change", "content_match", "state_persist"],
        "expected_scenarios": [
            sc("add_row", "เพิ่มแถว", "row", "added", "จำนวนแถว", "เพิ่มขึ้น", "new record"),
            sc("search_filters", "ค้นหา", "search", "กรอง", "filter", "แสดงเฉพาะ"),
            sc("delete_row", "ลบ", "delete", "หาย", "ลดลง", "removed"),
            sc("edit_row", "แก้ไข", "edit", "อัปเดต", "updated", "เปลี่ยน"),
        ],
    },
    # ───────────────────────────── orangehrm ─────────────────────────────
    {
        "page_id": "orangehrm_login", "site": "orangehrm",
        "url": "https://opensource-demo.orangehrmlive.com/web/index.php/auth/login", "title": "OrangeHRM — Login",
        "pam_content": "HR portal login: Username and Password textboxes, Login button. Admin/admin123 reaches "
                       "the Dashboard. Invalid credentials show 'Invalid credentials'. Empty fields show 'Required'.",
        "requires_auth": False,
        "elements": [
            el("textbox", "Username", True),
            el("textbox", "Password", True),
            el("button", "Login", True, "https://opensource-demo.orangehrmlive.com/web/index.php/dashboard/index"),
        ],
        "expected_test_count": 5,
        "expected_assertion_types": ["redirect", "validation_error", "content_match"],
        "expected_scenarios": [
            sc("valid_login", "dashboard", "แดชบอร์ด", "เข้าสู่ระบบ", "สำเร็จ"),
            sc("invalid_login", "invalid credentials", "ไม่ถูกต้อง", "ข้อมูลไม่ถูกต้อง", "error"),
            sc("empty_required", "required", "ต้องกรอก", "จำเป็น", "ว่าง"),
        ],
    },
    # ───────────────────────── automationexercise ─────────────────────────
    {
        "page_id": "automationexercise_signup", "site": "automationexercise",
        "url": "https://automationexercise.com/login", "title": "Automation Exercise — Login / Signup",
        "pam_content": "Combined page. New User Signup: Name and Email + Signup button. Login: Email and Password "
                       "+ Login button. A new email proceeds to account creation; existing email shows 'Email Address "
                       "already exist!'; wrong login shows 'Your email or password is incorrect!'.",
        "requires_auth": False,
        "elements": [
            el("textbox", "Signup Name", True),
            el("textbox", "Signup Email", True),
            el("button", "Signup", True),
            el("textbox", "Login Email", True),
            el("textbox", "Login Password", True),
            el("button", "Login", True),
        ],
        "expected_test_count": 6,
        "expected_assertion_types": ["redirect", "validation_error", "content_match"],
        "expected_scenarios": [
            sc("valid_signup", "account information", "สร้างบัญชี", "สมัครสมาชิก", "signup", "enter account"),
            sc("existing_email", "already exist", "มีอยู่แล้ว", "ซ้ำ", "อีเมลนี้"),
            sc("invalid_login", "email or password is incorrect", "ไม่ถูกต้อง", "ผิดพลาด"),
        ],
    },
    {
        "page_id": "automationexercise_contact", "site": "automationexercise",
        "url": "https://automationexercise.com/contact_us", "title": "Automation Exercise — Contact Us",
        "pam_content": "Contact form: Name, Email, Subject, Message textboxes, optional file upload, Submit. "
                       "Valid submission shows 'Success! Your details have been submitted successfully.' An invalid "
                       "email triggers browser validation; empty required fields block submission.",
        "requires_auth": False,
        "elements": [
            el("textbox", "Name", True),
            el("textbox", "Email", True),
            el("textbox", "Subject", True),
            el("textbox", "Message", True),
            el("button", "Submit", True),
        ],
        "expected_test_count": 5,
        "expected_assertion_types": ["content_match", "validation_error"],
        "expected_scenarios": [
            sc("valid_submit", "success", "ส่งสำเร็จ", "details have been submitted", "เรียบร้อย"),
            sc("invalid_email", "อีเมล", "email", "ไม่ถูกต้อง", "invalid"),
            sc("empty_required", "required", "ต้องกรอก", "ว่าง", "จำเป็น"),
        ],
    },
    # ─────────────────── realworld.habsida.net (Conduit) ───────────────────
    {
        "page_id": "habsida_login", "site": "habsida-realworld",
        "url": "https://realworld.habsida.net/login", "title": "Conduit — Sign in",
        "pam_content": "RealWorld/Conduit sign-in: Email and Password textboxes, Sign in button. Valid credentials "
                       "redirect to the home feed with the user's nav menu. Invalid credentials show 'email or "
                       "password is invalid'.",
        "requires_auth": False,
        "elements": [
            el("textbox", "Email", True),
            el("textbox", "Password", True),
            el("button", "Sign in", True, "https://realworld.habsida.net/"),
        ],
        "expected_test_count": 5,
        "expected_assertion_types": ["redirect", "validation_error", "content_match"],
        "expected_scenarios": [
            sc("valid_login", "home", "หน้าแรก", "feed", "เข้าสู่ระบบ", "สำเร็จ", "นำทาง"),
            sc("invalid_login", "is invalid", "ไม่ถูกต้อง", "error", "ผิดพลาด"),
            sc("empty_login", "required", "ต้องกรอก", "ว่าง", "blank"),
        ],
    },
    {
        "page_id": "habsida_register", "site": "habsida-realworld",
        "url": "https://realworld.habsida.net/register", "title": "Conduit — Sign up",
        "pam_content": "RealWorld/Conduit sign-up: Username, Email, Password textboxes, Sign up button. A valid "
                       "sign-up logs the user in and redirects home. A taken username/email shows 'has already been "
                       "taken'. An invalid email is rejected.",
        "requires_auth": False,
        "elements": [
            el("textbox", "Username", True),
            el("textbox", "Email", True),
            el("textbox", "Password", True),
            el("button", "Sign up", True, "https://realworld.habsida.net/"),
        ],
        "expected_test_count": 5,
        "expected_assertion_types": ["redirect", "validation_error", "state_persist"],
        "expected_scenarios": [
            sc("valid_register", "home", "สมัครสำเร็จ", "เข้าสู่ระบบ", "redirect", "หน้าแรก"),
            sc("taken_username", "already been taken", "ถูกใช้แล้ว", "ซ้ำ", "มีอยู่แล้ว"),
            sc("invalid_email", "อีเมล", "email", "ไม่ถูกต้อง", "invalid"),
        ],
    },
]


def main() -> None:
    out = Path(__file__).parent / "golden_set.jsonl"
    assert len(PAGES) == 20, f"expected 20 pages, got {len(PAGES)}"
    seen = set()
    with out.open("w", encoding="utf-8") as f:
        for p in PAGES:
            assert p["page_id"] not in seen, f"duplicate page_id {p['page_id']}"
            seen.add(p["page_id"])
            assert p["elements"], f"{p['page_id']} has no elements"
            assert p["expected_scenarios"], f"{p['page_id']} has no scenarios"
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(PAGES)} pages -> {out}")
    print(f"total expected_test_count = {sum(p['expected_test_count'] for p in PAGES)}")
    print(f"total expected_scenarios  = {sum(len(p['expected_scenarios']) for p in PAGES)}")


if __name__ == "__main__":
    main()
