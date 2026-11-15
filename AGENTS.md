# Project Rules

- Manage the Python environment with `uv`. Do not add runtime dependencies other than `numpy` (use Python stdlib such as `tkinter` for visualization).
- Keep comments minimal (only in non-obvious places) and do not add section-header comments.
- Ensure `ruff check .`, `ruff format --check .`, and `ty check` pass after changes.
