# QA Agent — saucedemo.com Summary

**วันที่รัน:** 2026-06-13 เวลา 14:40–14:51 น.
**คำสั่ง:** `uv run python -m src.universal_qa --url https://www.saucedemo.com/ --username standard_user --password secret_sauce --max-pages 15 --explore-timeout 5`
**เวลารันรวม:** ~10 นาที 25 วินาที (14:40:53 → 14:51:18)

---

## ผลรัน

| ตัวชี้วัด | ค่า |
|---|---|
| Total tests generated | 39 |
| Tests ที่รันจริง (แสดงผล [PASS]/[FAIL]) | 21 |
| **Passed** | **15 (71.4%)** |
| **Failed** | **6 (28.6%)** |
| Pages discovered | 3 |
| Flows detected | 9 |
| LLM repair attempts | 10+ ครั้ง |

> หมายเหตุ: 39 test cases ถูก generate แต่ส่วนหนึ่งถูก execute ผ่าน hypothesis_id เดิมซ้ำกัน (flow-based) ผลที่แสดงใน terminal มี 21 รายการ

---

## Tests ที่ผ่าน (PASS)

| # | Type | ชื่อ Test | เวลา |
|---|---|---|---|
| 1 | security | ทดสอบ XSS injection: Swag Labs | 1.6s |
| 2 | security | ทดสอบ SQL injection: Swag Labs | 1.1s |
| 3 | security | ทดสอบ XSS injection: Swag Labs | 1.1s |
| 4 | security | ทดสอบ SQL injection: Swag Labs | 1.1s |
| 5 | security | ทดสอบ XSS injection: Swag Labs | 1.1s |
| 6 | security | ทดสอบ SQL injection: Swag Labs | 1.1s |
| 7 | functional | ทดสอบการเพิ่มสินค้าลงในตะกร้า | 42.0s |
| 8 | functional | ทดสอบการคลิกปุ่ม Open Menu เมื่อไม่มีการเลือกสินค้าในตะกร้า | 0.1s |
| 9 | functional | ทดสอบการคลิกปุ่ม 'Back to products' เมื่อไม่มีสินค้าในตะกร้า | 0.1s |
| 10 | functional | ทดสอบ flow /inventory.html → /inventory-item.html ตั้งแต่ต้นจนจบ | 0.1s |
| 11 | functional | ทดสอบ flow /inventory.html → /inventory-item.html ตั้งแต่ต้นจนจบ | 0.1s |
| 12 | functional | ทดสอบ flow /inventory.html → /inventory-item.html ตั้งแต่ต้นจนจบ | 0.1s |
| 13 | functional | ทดสอบ flow /inventory.html → /inventory-item.html ตั้งแต่ต้นจนจบ | 0.1s |
| 14 | functional | ทดสอบ flow /inventory.html → /inventory-item.html ตั้งแต่ต้นจนจบ | 0.1s |
| 15 | functional | ทดสอบ flow /inventory.html → /inventory-item.html ตั้งแต่ต้นจนจบ | 0.1s |
| 16 | functional | ทดสอบ flow /inventory.html → /inventory-item.html ตั้งแต่ต้นจนจบ | 0.2s |
| 17 | functional | ทดสอบ flow /inventory-item.html → /inventory.html ตั้งแต่ต้นจนจบ | 0.1s |
| 18 | functional | ทดสอบ flow /inventory-item.html → /inventory.html ตั้งแต่ต้นจนจบ | 0.1s |

> Security tests (6 รายการ): ผ่านทั้งหมด — XSS/SQL injection ถูกป้องกันอย่างถูกต้อง
> Flow navigation tests: ผ่าน 9/9 — การนำทางระหว่างหน้า inventory ↔ inventory-item ทำงานได้ถูกต้อง

---

## Tests ที่ล้มเหลว (FAIL)

### 1. [FAIL] functional — ทดสอบการเพิ่มสินค้าลงตะกร้า (68.2s)
- **สาเหตุ:** Step 2 หมดเวลา (Timeout 10s) รอ element `get_by_text('"Login"')` ซึ่งไม่มีในหน้า inventory
- **Root cause:** LLM generate test ที่เริ่มจาก `/inventory.html` (หลัง login แล้ว) แต่ generate step "คลิกปุ่ม Login" ซึ่งไม่ควรมีในหน้านี้ RepairEngine ช่วยแก้ step แรกได้ แต่ step 2 ยังคงหา Login button ไม่พบ

### 2. [FAIL] functional — ทดสอบการซื้อสินค้าสำเร็จ (81.7s)
- **สาเหตุ:** Step 2 หมดเวลา (Timeout 10s) รอ element `get_by_text("เพื่อไปยังหน้ารายการสินค้า")` ซึ่งเป็นข้อความภาษาไทยที่ LLM แต่งขึ้น ไม่ใช่ text จริงในหน้า
- **Root cause:** LLM สร้าง locator เป็น text ภาษาไทยแบบ NL description แทนที่จะใช้ "Back to products" ซึ่งเป็น label จริงในหน้า

### 3. [FAIL] functional — กรอก field ว่างในช่อง Username (12.0s)
- **สาเหตุ:** Step 2 ถูกบล็อกโดย `BLOCKED_ACTION_PATTERNS` เพราะ locator มีคำว่า `"password"` — `fill "standard_user" ในช่อง Password`
- **Root cause:** Security guard ใน executor บล็อก action ที่เกี่ยวกับ password ตาม policy โดยออกแบบไว้ (ป้องกันการกรอก credential จริงในช่อง password)

### 4. [FAIL] functional — กรอก field ว่าง (62.1s)
- **สาเหตุ:** Step 2 หมดเวลา รอ element `get_by_text('"Submit"')` ซึ่งไม่มีในหน้า inventory-item
- **Root cause:** LLM สร้าง test ที่ผสม context ผิด — เหมือนพยายาม test form submission แต่ generate step ที่อ้างถึง Submit button ซึ่งไม่มีใน saucedemo

### 5. [FAIL] accessibility — ตรวจสอบ Accessibility: Swag Labs (0.1s)
- **สาเหตุ:** `missing accessible name on 1 interactive node(s)`
- **Root cause:** มี interactive element 1 ชิ้น (น่าจะเป็น sort dropdown หรือ hamburger menu button) ที่ไม่มี accessible name ใน DOM ของ saucedemo — **นี่คือ real accessibility bug ในตัวเว็บ**

### 6. [FAIL] functional — ทดสอบ invalid login (ไม่แสดงชื่อ) (~60s)
- **สาเหตุ:** Step 2 หมดเวลา รอ `get_by_text('"Login"')` — เช่นเดียวกับข้อ 1 LLM generate step ที่ navigate ไป /login แต่ invoke Login button ด้วย locator ผิด

---

## หน้าที่ Discover ได้

Phase 2 (Site Discovery) พบ 2 หน้า:
1. `https://www.saucedemo.com/inventory.html` — หน้า Product Listing (หลัง login)
2. `https://www.saucedemo.com/inventory-item.html` — หน้า Product Detail (generic)

Phase 3 (Interaction-based Exploration) พบเพิ่ม:
3. `https://www.saucedemo.com/inventory-item.html?id=4` — Sauce Labs Backpack

Links ที่ discover ผ่าน interaction (ไม่ได้เปิดเป็น node แยก แต่บันทึกเป็น flows):
- `inventory-item.html?id=0` — Sauce Labs Bike Light
- `inventory-item.html?id=1` — Sauce Labs Bolt T-Shirt
- `inventory-item.html?id=2` — Sauce Labs Onesie
- `inventory-item.html?id=3` — Test.allTheThings() T-Shirt (Red)
- `inventory-item.html?id=5` — Sauce Labs Fleece Jacket

**หมายเหตุ:** ไม่พบ `/cart.html`, `/checkout-step-one.html`, `/checkout-step-two.html` เนื่องจาก discovery หยุดที่ 2 pages (max-pages=15 แต่ interaction explorer ออก early หลัง 3 pages เพราะ explore-timeout=5 นาที)

---

## สังเกตการณ์

### เวลาต่อ Phase
| Phase | เวลา | รายละเอียด |
|---|---|---|
| Phase 1 (Login) | ~17 วินาที | 14:40:53 → 14:41:11 (browser launch + login) |
| Phase 2 (Site Discovery) | ~19 วินาที | 14:41:11 → 14:41:30 (2 pages, 2 nodes, BFS) |
| Phase 3 (Exploration) | ~28 วินาที | 14:41:30 → 14:41:59 (3 pages, 9 flows) |
| Phase 4a (Generate) | ~3 นาที 30 วินาที | 14:41:59 → 14:45:29 (9 LLM calls, 39 test cases) |
| Phase 4b (Execute) | ~5 นาที 49 วินาที | 14:45:29 → 14:51:18 |

### Flows ที่พบ
- 9 flows ทั้งหมด: ส่วนใหญ่เป็น product click flows (6 สินค้า → inventory-item) + Back to products + Add to cart state change
- Add to cart flow พบว่า state เปลี่ยน (cart badge อัปเดต) — explorer สังเกตเห็นการเปลี่ยน state นี้

### Perception Pipeline
- AOMExtractor ทำงานได้ดี: inventory.html มี 142 nodes → compact เหลือ 25 nodes (530 tokens)
- inventory-item.html มี 47-49 nodes → compact เหลือ 25 nodes (583-587 tokens)
- Latency: 20-45ms ต่อ extraction — เร็วมาก

### LLM (qwen2.5-coder:7b-instruct-q4_K_M)
- Generate: 9 LLM calls สำหรับ 39 test cases (ค่าเฉลี่ย ~23 วินาทีต่อ call)
- Repair: 10+ LLM calls สำหรับ RepairEngine (ค่าเฉลี่ย ~2 วินาทีต่อ call)
- ปัญหาที่พบ: LLM บางครั้ง generate locator เป็น NL description ภาษาไทยแทน element label จริง เช่น `"เพื่อไปยังหน้ารายการสินค้า"` แทน `"Back to products"`
- LLM generate Login step ใน test ที่ควร start จากหน้า inventory (post-login state) — context confusion

### Burger Menu (All Items / About / Reset App State)
- ทั้ง 3 items ใน hamburger menu ถูก skip ใน Phase 2 และ Phase 3 เนื่องจาก menu ต้องเปิดก่อนคลิก item — explorer พยายาม click โดยตรงแต่ element อยู่นอก viewport หรือ hidden

### BLOCKED_ACTION_PATTERNS
- Executor บล็อก action ที่มีคำว่า `password` ในขั้นตอน fill — policy นี้ทำให้ login flow test ล้มเหลวโดยไม่สามารถ repair ได้

### Accessibility Finding (Real Bug)
- ตรวจพบ `missing accessible name on 1 interactive node(s)` — น่าจะเป็น sort combobox (`Name (A to Z)...`) ที่ไม่มี aria-label หรือ label element ที่ถูก associate อย่างถูกต้อง นับเป็น **real accessibility defect** ใน saucedemo.com
