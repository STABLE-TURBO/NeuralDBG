# Distribution Journal

## 2025-10-16
- Features added:
  - `neural docs` command (Markdown; PDF via Pandoc if available)
  - Auto-flatten policy flag for Dense/Output; strict by default
- Bugs fixed:
  - TensorFlow codegen now applies layers to `x` (e.g., `Dense(...)(x)`)
  - Output-on-4D inputs now enforced (error by default; optional flatten)
- Tests added:
  - TF Dense is-applied test
  - Output-on-4D strict vs. auto-flatten tests
  - Docgen smoke test
- Packaging groundwork:
  - Added `pyproject.toml` (build isolation)
  - Updated setup.py version to `0.3.0.dev0`
  - Version detection now tries `neural-dsl` then `neural`, fallback `0.3.0-dev`


- Environment setup:
  - Installed optional dev dependencies: jax[cpu], optuna (for version and CI checks)
- Repo hygiene:
  - Added ROADMAP.md to .gitignore (keep roadmap local, reduce distribution noise)

- CI skeleton:
  - Added .github/workflows/ci.yml (Ubuntu+Windows, Python 3.11): ruff, mypy, pytest, pip-audit (non-blocking)
- Static analysis config:
  - Added ruff config in pyproject.toml
  - Added mypy.ini with permissive baseline (to tighten incrementally)

- README updates:
  - Added CI badge and Reproducibility section (seed utility usage)
- CI enhancement:
  - Added nightly schedule (02:00 UTC) to CI workflow
- Tests:
  - Added tests/code_generator/test_policy_and_parity.py for flatten policy and TF/PT parity
