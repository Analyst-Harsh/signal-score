# Candidate models — plan overview

## Context

Production today is a single, never-really-tested model:
`TF-IDF(word 1-2gram) + TF-IDF(char_wb 3-5gram) + is_member_plus → LogisticRegression`,
PR-AUC 0.5653 / minority-F1 0.4824 / Brier 0.1829 (`EXPERIMENTS.md`, 2026-08-29). It became
"production" only via the first-promotion floor (`evaluation/run_gate.py:46`,
`FIRST_PROMOTION_MIN_PR_AUC = 0.5653`, sourced directly from this same run) — there was no
prior production model to beat, so it has never actually won a real head-to-head margin
test.

The goal now is to evaluate alternatives — richer features, XGBoost, BGE embeddings — while
holding the repo to the same production discipline throughout: every candidate goes through
the real promotion gate (unit tests → I/O contract → head-to-head margin vs. current
production on the frozen `eval_set_v1` set), and every result, win or loss, gets an honest
row in `EXPERIMENTS.md`.

The initial instinct was to sequence embeddings first: (1) embeddings + LogReg, (2) XGBoost
+ the existing TF-IDF vectorizer, (3) XGBoost + embeddings. Before committing to that order,
three independent senior-ML-engineer consultations (applied-NLP lens, MLOps lens,
mentorship/pedagogy lens) were run against the actual codebase to sanity-check it. All three
converged independently on the same finding and the same reordering below.

### The finding that reordered the plan

`features/schema.py`'s `FeatureRow` (the frozen serving contract, `extra="forbid"`,
`frozen=True`) already declares 11 leakage-safe text-signal columns — `title_chars`,
`body_chars`, `body_words`, `has_code_block`, `has_stack_trace`, `has_url`, `has_logline`,
`has_excl`, `code_ratio`, `any_lex`, `is_ci_flake_shaped`. They're computed by
`features/pipeline.py:build_features()`, DVC'd, EDA'd (mostly monotonic across priority
classes) — but that same function's `TextDiagnostics` bundle is explicitly commented "never
concatenated into the model matrix," and `training/train_baseline.py:build_feature_matrix()`
(lines 82-95) confirms it: only `word_matrix, char_matrix, member_column` get hstacked. All
11 columns are silently dropped at training time. **The current baseline isn't even using
signal already sitting in its own contract.**

### Why the original order was reconsidered

Embeddings-first changes two variables at once (feature source *and* model family) in most
steps, so a win is uninterpretable — you can't tell what caused it. It also front-loads the
highest-risk, highest-cost item (BGE: a new ~2GB dependency, an extension to the frozen
`FeatureRow`/`build_features()` contract used by both training and the live `/score`
endpoint, a real risk against the promotion gate's 2.0s latency bound
(`evaluation/contract.py:_MAX_PREDICT_LATENCY_SECONDS`), and a jump from 12 interpretable
Evidently drift dimensions to a 768-dim problem) before a single real gate cycle has been
run. XGBoost on raw sparse TF-IDF (100k+ columns) is also a known weak matchup for trees vs.
a linear model — XGBoost only gets a fair fight on dense features.

## Decided sequence

Four experiments, one variable changed per step, every candidate through the real
promotion gate, every result logged in `EXPERIMENTS.md`:

1. **LogReg + TF-IDF + the 11 already-declared-but-unused diagnostics.** Pure feature-source
   delta vs. current production. Zero new dependencies, zero contract change, zero latency
   risk — the cheapest possible real head-to-head the gate has ever run.
2. **Hyperparameter sweep on the current production feature set.** The 11 diagnostics are
   excluded entirely — step 1's real gate result already rejected them as a block, and that
   result is treated as conclusive, not reopened via a per-feature necessity check. Tuned on
   exactly the TF-IDF(word) + TF-IDF(char) + `is_member_plus` -> `LogisticRegression` feature
   set, on the existing train→val split directly — no k-fold CV needed, since the project
   already has a purpose-built val split reserved for exactly this, consistent with
   `train_baseline.py`'s existing fit-on-train/transform-on-val discipline. Sweeps `C`,
   `penalty` (L1 vs. L2), word n-gram range, word `min_df`, and word `sublinear_tf` — the char
   vectorizer stays fixed at its locked-spec values to keep the grid tractable. The sweep
   (`tune_baseline.py`) is search-only — it never saves a model or touches the registry;
   it prints a ranked trial table and the exact `train_baseline.py` command that reproduces
   the winning config as the one real, gated candidate. Gate that candidate.
3. **XGBoost on the tuned winner of (2)'s feature set** — `is_member_plus` plus a dense
   representation of the TF-IDF text signal (e.g. SVD-compressed) rather than raw sparse
   n-grams, since XGBoost is a weak matchup for trees on 100k+ sparse columns. Pure
   model-family delta, isolates whether non-linearity/interactions help at all on this data.
4. **BGE embeddings — only after (1)-(3)** establish an honest, tuned, best-available
   classical baseline. Budgeted as its own infrastructure change (new dependency extra,
   pinned model revision, startup warm-load, an explicit latency assertion in the contract
   test) and evaluated separately from its own head-to-head result, since it's the only step
   that touches the frozen `FeatureRow` contract and the live serving path.

## Explicitly deferred

This document captures the decided sequence and rationale only — it is not a file-by-file
implementation plan. Implementation details for each experiment (exact code changes,
`EXPERIMENTS.md` rows, gate runs) are scoped separately, one experiment at a time, starting
with experiment 1.
