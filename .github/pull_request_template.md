## Summary

<!-- What changed and why. Focus on the why -- the diff already shows the what. -->

## Type of change

- [ ] Pipeline code (`collection` / `features` / `training` / `evaluation` / `serving` / `monitoring`)
- [ ] New or updated training run (real numbers only -- see below)
- [ ] Data / dataset change (raw fetch, split, label mapping)
- [ ] Docs / design (`docs/`, `CLAUDE.md`)
- [ ] Tooling / CI / dependencies

## Test plan

- [ ] `make check` passes (lint, format-check, pyright strict, test+coverage)
- [ ] Tests added in this PR for any new/changed logic (no upfront empty test dirs)
- [ ] If this touches `features/` or `training/split.py`: no leakage introduced -- fit-on-train / transform-on-val still holds structurally, not just by convention

## If this is a real training run

- [ ] Numbers are from an actual run, not estimated or fabricated (CLAUDE.md hard rule)
- [ ] A row was added to `EXPERIMENTS.md`, including if it's a loss
- [ ] Model only gets tagged `production` after all three promotion-gate steps pass (unit tests -> I/O contract -> head-to-head margin win vs. current production on the frozen eval set) -- never hand-flipped
- [ ] No `priority/*`-taxonomy training data blended with a different label taxonomy (holdout repos stay zero-shot only)

## Design check

- [ ] New classes, if any, justified against `docs/design-patterns-guide.md` (a class needs a real, currently-used extension point -- not "might need it later")

<!-- 🤖 If this PR was authored or co-authored by Claude Code, leave the session link below. -->
