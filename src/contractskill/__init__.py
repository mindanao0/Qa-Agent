"""contractskill package.

Backward-compatibility re-exports.

The audit referenced ``src/contractskill/grounder.py``, but that file never
existed — the Grounder lives in :mod:`src.perception.grounder`. This re-export
lets callers do ``from src.contractskill import Grounder`` regardless of which
path they expect, without duplicating the implementation.
"""
from src.perception.grounder import Grounder

__all__ = ["Grounder"]
