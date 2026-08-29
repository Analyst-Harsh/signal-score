# CLAUDE.md

## One-liner

Signal Score: classical ML service that scores GitHub issue priority from
GitHub metadata, text signals, and BGE embeddings.

## Source of truth

Architecture and rules live in `docs/signalscore-design.md` and
`docs/signalscore-architecture.mermaid`. Read them before touching pipeline
logic — this file doesn't duplicate them, it points at them.

Class-based design and Gang-of-Four pattern usage are governed by
`docs/design-patterns-guide.md` — read it before choosing between a function
and a class for new code.

## Module map

| Module | Role |
|---|---|
| `collection` | Data source stage — pulls GitHub issues (K8s primary, k8s-sigs volume supplement, cross-taxonomy holdout) |
| `features` | Feature pipeline — text signals, GitHub metadata, BGE embeddings |
| `training` | Experimentation — split, sklearn/XGBoost configs, W&B tracking, registers "staging" |
| `evaluation` | The promotion gate — unit/contract checks, head-to-head margin eval, tag flip |
| `serving` | FastAPI app exposing `/score`, loads the "production"-tagged model |
| `monitoring` | Evidently AI drift monitoring on live traffic |
| `registry.py` | Shared MLflow client used by `training`, `evaluation`, and `serving` |

## Commands

- `make install` — sync deps, install lefthook git hooks
- `make check` — lint + format-check + typecheck + test (run this before considering any task done)
- `make lint` — `ruff check`
- `make format` — `ruff format`
- `make typecheck` — `pyright` (strict)
- `make test` — `pytest` with coverage

## Conventions

- **uv only** — no bare `pip`/`venv`.
- **src layout** — package code lives under `src/signalscore/`.
- Core dependencies (`requests`, `pandas`, `pydantic`, `python-dotenv`) are
  always installed; `ml` (scikit-learn, xgboost) and `serving` (fastapi,
  uvicorn) are opt-in extras.
- Tests are added in the same PR as the code they cover — no upfront empty
  test directories.
- Ruff handles lint, format, and import sort — no isort/black.
- Pyright strict handles type checking — no mypy.
- Lefthook handles git hooks — no pre-commit.

## Data rules (hard guardrail)

Never blend `priority/*`-taxonomy training data with a different label
taxonomy (e.g. Rust's `P-high/P-medium/P-low`, VS Code's severity labels).
Those repos are zero-shot generalization holdout only — never trained on.

## Promotion gate rule

A model only gets the `production` MLflow tag after passing all three gate
steps: feature-pipeline unit tests → model I/O contract test → head-to-head
margin win vs. current production on the frozen held-out set. Never
hand-flip a tag. Losses get logged in `EXPERIMENTS.md`, not discarded.

## What NOT to build yet

- No TriageBot integration.
- No priority-queue/dispatch refactor.
- No Terraform/AWS infra.
- No Next.js dashboard code (lives in a separate repo).
- No Dockerfile until `serving/` has real code.

## EXPERIMENTS.md

Every real training run, including losses, gets a row in `EXPERIMENTS.md`.
No fabricated numbers.
