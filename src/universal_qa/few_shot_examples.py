"""Few-shot exemplars for the Universal test planner (Phase-2 of the eval track).

The base 7B writes a lot of shallow assertions ("หน้าเว็บตอบสนอง...ได้ถูกต้อง").
These three exemplars are distilled from the highest-quality, concrete-assertion
examples in data/training/train.jsonl (quality=1.0) — one per assertion family
the golden set rewards:

  1. specific-URL redirect   (navigation reaches a named page)
  2. validation message      (bad input -> a concrete error string)
  3. state / count change     (an action changes observable state)

They are injected into the planner's per-element prompts so the model imitates
*checkable* outcomes instead of generic "responds correctly". Kept short to stay
within the 7B context budget and to not dominate the page-specific instruction.
"""
from __future__ import annotations

# Rendered verbatim into the functional / negative / edge prompts.
FEW_SHOT_BLOCK = (
    "ตัวอย่างการเขียน test case ที่ดี (assertion ตรวจ state จริง ไม่ใช่แค่ \"แสดงผล\"):\n"
    "\n"
    "ตัวอย่าง 1 — Element: button \"Sign in\" (นำไปยังหน้า dashboard)\n"
    "{\"test_cases\": [{\"title\": \"เข้าสู่ระบบด้วยข้อมูลที่ถูกต้องแล้ว redirect ไป dashboard\", "
    "\"priority\": \"high\", \"preconditions\": [\"มีบัญชีผู้ใช้ที่ถูกต้อง\"], "
    "\"steps\": [\"กรอก \\\"Username\\\" ด้วย standard_user\", \"กรอก \\\"Password\\\" ด้วย secret_sauce\", "
    "\"คลิกปุ่ม \\\"Sign in\\\"\", \"ตรวจสอบว่า URL เปลี่ยนไปหน้า dashboard\"], "
    "\"expected_outcome\": \"ระบบ redirect ไปยังหน้า dashboard และแสดงเมนูของผู้ใช้ที่ล็อกอิน\"}]}\n"
    "\n"
    "ตัวอย่าง 2 — Element: textbox \"Email\"\n"
    "{\"test_cases\": [{\"title\": \"กรอกอีเมลผิดรูปแบบแล้วระบบแจ้ง error\", "
    "\"priority\": \"medium\", \"preconditions\": [\"อยู่ที่หน้าฟอร์ม\"], "
    "\"steps\": [\"กรอก \\\"Email\\\" ด้วย not-an-email\", \"คลิกปุ่ม \\\"Submit\\\"\", "
    "\"ตรวจสอบข้อความ validation ใต้ช่อง Email\"], "
    "\"expected_outcome\": \"แสดงข้อความ error ว่าอีเมลไม่ถูกต้อง และฟอร์มไม่ถูกส่ง\"}]}\n"
    "\n"
    "ตัวอย่าง 3 — Element: button \"Add to cart\"\n"
    "{\"test_cases\": [{\"title\": \"คลิก Add to cart แล้วจำนวนสินค้าในตะกร้าเพิ่มขึ้น\", "
    "\"priority\": \"high\", \"preconditions\": [\"อยู่ที่หน้ารายการสินค้า\"], "
    "\"steps\": [\"จดจำนวนสินค้าในตะกร้าปัจจุบัน\", \"คลิกปุ่ม \\\"Add to cart\\\"\", "
    "\"ตรวจสอบตัวเลข badge บนไอคอนตะกร้า\"], "
    "\"expected_outcome\": \"badge จำนวนสินค้าในตะกร้าเพิ่มขึ้น 1 (เช่น จาก 0 เป็น 1)\"}]}\n"
)


def prepend_few_shot(prompt: str) -> str:
    """Return `prompt` with the few-shot block prepended."""
    return f"{FEW_SHOT_BLOCK}\n{prompt}"


__all__ = ["FEW_SHOT_BLOCK", "prepend_few_shot"]
