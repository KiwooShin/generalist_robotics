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
- Plan the `src/` layout before writing code: sketch the package tree
  (subpackages, responsibilities, dependency direction) first.
- Never keep all source files in a single flat level — group them into
  subpackages by responsibility (e.g. `data/`, `models/`, `training/`, `envs/`,
  `utils/`, `cli/`), each with an `__init__.py` and one clear job.
- Well-managed package layout: small modules with one responsibility each;
  split a subpackage once it grows past ~5 files with distinct concerns.
- Tests with `unittest` for every module, named `test_<module>.py` and kept in
  a `tests/` folder sibling to the module (`src/foo/bar.py` →
  `src/foo/tests/test_bar.py`), runnable via `python -m unittest discover`.

## Documents
- `research.md` — research paper summaries (living survey).
- `plan.md` — project plan and roadmap.
- `progress/YYYY-MM-DD.md` — daily progress log, one file per day.

## Git
- Commit as frequently as possible — each meaningful change is its own commit
  (GitHub activity matters). Push after committing.
- Every milestone ships a demo video/GIF in the README.
