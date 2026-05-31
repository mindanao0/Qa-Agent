from src.llm.structured import enforce_json_output, TestPlan
import json

# Simulate a raw LLM response with markdown fence
raw = '```json\n{\n  "title": "Login test",\n  "requirement_summary": "Verify login button is visible",\n  "estimated_complexity": "low",\n  "steps": [{"step_number": 1, "description": "Navigate", "action": "goto", "expected_result": "page loads"}]\n}\n```'

plan = enforce_json_output(TestPlan, raw)
assert isinstance(plan, TestPlan)
assert plan.title == 'Login test'
print('Legacy path OK, title:', plan.title)
