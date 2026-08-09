# Coding Rules

Conventions for all code in this repo.

## Style
- Follow standard Python conventions (PEP 8): `snake_case` functions/variables,
  `PascalCase` classes, type hints on public APIs.
- Production-level code: clear structure, no dead code, no quick hacks left behind.
- Comments: short and sparse — only where the code cannot speak for itself.
- Every file starts with a 1-line file docstring.
- Every class has a docstring. Public functions get a short docstring too.
- Avoid private functions/classes (`_name`). Prefer small public functions with clear names.

## Structure
- Well-managed package layout: small modules with one responsibility each.
- Tests with `unittest` for every module, kept under `tests/`, runnable via
  `python -m unittest discover`.

## Documents
- `research.md` — research paper summaries (living survey).
- `plan.md` — project plan and roadmap.
- `progress/YYYY-MM-DD.md` — daily progress log, one file per day.

## Git
- Commit as frequently as possible — each meaningful change is its own commit
  (GitHub activity matters). Push after committing.
- Every milestone ships a demo video/GIF in the README.
