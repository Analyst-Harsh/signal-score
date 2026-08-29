# Experiments

Every real training run gets a row — including losses. The promotion gate
saying "no" is the point, not a bug.

## Week 1 — Dataset assembly & split (2026-08-29, snapshot `eval-set-v1`)

Source: `data/raw/kubernetes-kubernetes` (12,025 raw rows across the 3
label-queried JSONL files) → `training/build_dataset.py` → DVC-tracked
`data/processed/kubernetes-kubernetes` + `data/splits/kubernetes-kubernetes`.

**Drop report** (`DropReport`, recomputed from the raw corpus, not cached):

| Stage | Rows | Delta |
|---|---|---|
| Raw rows (3 files, pre-dedup) | 12,025 | — |
| Unique issues (post-dedup) | 11,991 | −34 duplicate fetches (same issue, multiple `priority/*` queries) |
| Dropped: multi-priority conflict | | −34 (issue currently carries ≥2 `priority/*` labels, ambiguous) |
| Dropped: `priority/important-longterm` (excluded) | | −56 |
| Dropped: `priority/awaiting-more-evidence` (excluded) | | −2 |
| **Kept** | **11,899** | 99.2% of unique issues retained |

**Split** (time-based frozen test, hash-based train/val on the rest):

| Split | Rows | Share |
|---|---|---|
| train | 7,763 | 65.2% |
| val | 1,756 | 14.8% |
| test (frozen, most-recent 20% by `created_at`) | 2,380 | 20.0% |
| **Total** | **11,899** | 100% |

Matches the design target (~65/15/20) exactly. `test` is frozen at git tag
`eval-set-v1` and is never reshuffled; `train`/`val` are hash-bucketed
(`VAL_BUCKET_CUTOFF = 19`) on the remaining 80%.

## Week 1 — EDA (2026-08-29, no model trained yet)

Computed on the full assembled matrix (`features_v1.jsonl`, 11,899 rows,
`eval-set-v1` snapshot), not just train — this is a class/feature-shape
check, not a train-only metric.

**Class balance:** P1_soon 40.9% / P2_backlog 40.5% / P0_critical 18.6% —
matches the majority-class-inflation number already cited in
`docs/priority-signal-decision.md`, confirming the assembled matrix reflects
the same distribution the split-strategy decision was based on.

**Feature distributions per class** — text signals separate classes more
cleanly than the one metadata bit:

| Feature (mean unless noted) | P0_critical | P1_soon | P2_backlog |
|---|---|---|---|
| body_chars | 2332 | 1856 | 1645 |
| code_ratio | 0.335 | 0.241 | 0.237 |
| has_code_block | 55.0% | 41.4% | 40.0% |
| has_stack_trace | 40.3% | 24.4% | 23.5% |
| has_url | 78.6% | 58.3% | 52.8% |
| has_logline | 10.8% | 8.1% | 3.9% |
| has_excl | 25.5% | 15.1% | 13.0% |
| any_lex | 38.6% | 32.9% | 28.4% |
| is_member_plus | 33.1% | 32.0% | 20.1% |

Nearly every diagnostic text flag is monotonically higher for P0 than P1
than P2 — longer, more code-heavy, more stack-trace-bearing reports skew
critical, which is the intuitive direction and a good sign these features
carry real signal. `is_ci_flake_shaped` is the one non-monotonic flag (P0
42.4% / P1 19.9% / P2 25.5%) — plausibly major CI infra breakages get
triaged critical while routine flakes get backlogged; not fully explained,
worth a closer look once a model's feature importances are in hand.

**Metadata/label correlation:** `is_member_plus` correlates weakly with
label (r=0.122, ordinal P2=0/P1=1/P0=2) — members/owners/collaborators file
relatively more P0/P1 (33.1%/32.0%) than non-members (20.1% for P2-only).
Real but weak alone, consistent with the design doc's choice to keep this a
single whitelisted bit rather than a broader metadata block.
`body_chars`/`body_words` are near-collinear (r=0.934, expected — same
underlying quantity) — worth remembering during feature selection so
neither is mistaken for two independent signals. No feature reaches even
r=0.15 with label individually; that's expected for a problem this needs a
model for, not a threshold rule.

## Log format

| Date | Model config | Dataset snapshot | PR-AUC | Minority F1 | Brier/ECE | Gate result | Notes |
|---|---|---|---|---|---|---|---|
| 2026-08-29 | baseline_v0: tfidf(word 1-2 + char_wb 3-5) + is_member_plus -> LogisticRegression | kubernetes-kubernetes | 0.5653 | 0.4824 | 0.1829 | n/a — no promotion gate yet | First real baseline run. accuracy=0.5564, macro_f1=0.5396 (clears the ≥46.8 macro-F1 target from `docs/priority-signal-decision.md`). Per-class F1: P0=0.482, P1=0.507, P2=0.629. `uv run python -m signalscore.training.train_baseline --train data/processed/kubernetes-kubernetes/train.jsonl --val data/processed/kubernetes-kubernetes/val.jsonl` |
