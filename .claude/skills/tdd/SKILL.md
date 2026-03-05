---
description: Enforce the TDD planning and verification cycle
---

# TDD Skill

## Command: /plan
When the user asks for a feature, first:
1. Create a markdown file in `docs/plans/<feature-name>.md`.
2. Define the exact function signature.
3. List 3-5 edge cases (empty input, wrong types, large files).
4. Do not write implementation code yet.

## Command: /verify
Perform a production-readiness check:
1. Run `uv run pytest`.
2. Run `uv run ruff check .`.
3. Run `uv run pyright`.
4. Check that all new functions have Google-style docstrings.
5. Provide a "Pass/Fail" summary for each.
