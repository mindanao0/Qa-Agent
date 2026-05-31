# src/perception/__init__.py
"""
Perception layer — Universal DOM Compression Pipeline.

Layer 1: AOMExtractor  — Playwright accessibility snapshot
Layer 2: DOMPruner     — Prune4Web browser-side scoring
Layer 3: SemanticCompactor — Groups into CompactPAM
Layer 4: Visual fallback — TODO Sprint 2
Entry:   Grounder       — Orchestrates all layers
"""
