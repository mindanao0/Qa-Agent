"""Sprint 11 — Continuous Mode.

Top-level continuous loop that wraps the Sprint 4/5/6/7 sub-graphs:
discover flows -> generate tests -> execute -> self-heal -> update SFG -> loop,
until a stop condition (max cycles / coverage plateau / memory limit).
"""
