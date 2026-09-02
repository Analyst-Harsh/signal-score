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
| 2026-08-31 | baseline_v1_diagnostics (candidate v6, run 4fa5b665346c4b54a85f661b73892e6b): tfidf(word 1-2 + char_wb 3-5) + is_member_plus + 11 diagnostics (min-max, clip=True) -> LogisticRegression | kubernetes-kubernetes | 0.4937 | 0.5636 | 0.2007 | fail | Experiment 1 (`docs/candidate_models/overview.md`): turns on the 11 diagnostic columns (`title_chars`, `body_chars`, `body_words`, `has_code_block`, `has_stack_trace`, `has_url`, `has_logline`, `has_excl`, `code_ratio`, `any_lex`, `is_ci_flake_shaped`) that `FeatureRow` already computed but `build_feature_matrix()` never fed to the model — min-max scaled (`clip=True`), fit on train only. Gated for real via `run_gate.py` against production v1 on `eval_set_v1`: REJECTED — pr_auc_macro improvement not significant at 95% CI (lower bound −0.0064 ≤ 0). accuracy=0.5139, macro_f1=0.4781. Per-class F1: P0=0.5636, P1=0.5382, P2=0.3325. Confirms `docs/v0-feature-selection.md`'s earlier informal ablation (−2.75 macro-F1 on an 80/20 split) with a real, bootstrapped gate result instead of an ad hoc script. Code reverted, not merged to main; production v1 unchanged. `uv run python -m signalscore.evaluation.run_gate --candidate-version 6 --notes "Experiment 1: TF-IDF + is_member_plus + 11 diagnostics (min-max, clipped) vs. production v1"` |
| 2026-08-31 | baseline_v0 hyperparameter sweep (Experiment 2, 64-cell grid) — winner is hyperparameter-identical to production: tfidf(word 1-2 min_df=3 sublinear_tf=True + char_wb 3-5) + is_member_plus -> LogisticRegression(C=1.0, penalty=l2) | kubernetes-kubernetes | 0.5653 | 0.4824 | 0.1829 | n/a — winner == current production config, no candidate to gate | Experiment 2 (`docs/candidate_models/overview.md`): full-factorial sweep (`tune_baseline.py`) over `C`∈{0.01, 0.1, 1, 10}, `penalty`∈{l2, l1}, word `ngram_range`∈{(1,1), (1,2)}, word `min_df`∈{3, 5}, word `sublinear_tf`∈{True, False} — 11 diagnostics excluded entirely per Experiment 1's rejection, not re-tested. No config in the grid beat production; the top result is hyperparameter-identical to it (winning metrics match the 2026-08-29 production row exactly). Strict win, not a tie — next-best was 0.5646 (`word_min_df=5`). Consistent pattern across all 64 cells: `l2` beats `l1` everywhere (best `l1` only 0.5442); `word_ngram_range=(1,2)`, `word_min_df=3`, and `word_sublinear_tf=True` each outperform their alternative at matched settings. The 8 cells at `C=0.01, penalty=l1` collapse to identical, degenerate results (pr_auc_macro=0.3512, minority_f1=0.0000) — that much L1 regularization zeroes every TF-IDF coefficient regardless of word-vectorizer settings, so the model stops predicting `P0_critical` at all. No new candidate produced, so no `run_gate.py` run — a config-identical winner has no real margin to gate against production. `uv run python -m signalscore.training.tune_baseline --train data/processed/kubernetes-kubernetes/train.jsonl --val data/processed/kubernetes-kubernetes/val.jsonl` |
| 2026-08-31 | xgboost_v1 (candidate v17, run f5ed9477e26d4e7cbf367926a918c837): tfidf(word 1-2 min_df=3 sublinear_tf=True + char_wb 3-5) -> TruncatedSVD(n_components=100, dense) + is_member_plus -> XGBClassifier(max_depth=6, learning_rate=0.01, subsample=0.6, colsample_bytree=0.6, min_child_weight=10, reg_alpha=1.0, reg_lambda=1.0, n_estimators=1000, early_stopping_rounds=30) | kubernetes-kubernetes | 0.4922 | 0.5560 | 0.1960 | fail | Experiment 3 (`docs/candidate_models/overview.md`): pure model-family delta on top of Experiment 2's tuned feature set — same TF-IDF(word 1-2 min_df=3 sublinear_tf=True + char_wb 3-5) + `is_member_plus` production config, but SVD-compressed to a 100-component dense matrix (rather than raw 100k+-column sparse n-grams) so XGBoost gets a fair matchup against trees. Timing calibration: one real `train_baseline.py --model-type xgboost` call measured 12.508s wall / 13.43s user CPU; budgeted the real sweep at `--budget 60` (below the tool's 80 default) to stay comfortably under a 20-minute ceiling given that estimate — the sweep then actually ran in 10:46 wall clock (821s user, 98s system, 142% CPU) for 60 random-sampled configs (`tune_xgboost.py`, full 7-knob grid is ~3,888 cells) over `max_depth`∈{2,3,4,6}, `learning_rate`∈{0.01,0.05,0.1,0.3}, `subsample`∈{0.6,0.8,1.0}, `colsample_bytree`∈{0.6,0.8,1.0}, `min_child_weight`∈{1,5,10}, `reg_alpha`∈{0.0,0.1,1.0}, `reg_lambda`∈{1.0,5.0,10.0}. Winner on `--val` (config above): pr_auc_macro=0.5411, minority_f1=0.4475, brier_macro=0.1871 — the real, gated candidate reproduced those exact val numbers bit-for-bit (accuracy=0.5245, macro_f1=0.5070, per-class F1 P0=0.4475/P1=0.4771/P2=0.5964), registered as MLflow candidate v17. Gated for real via `run_gate.py` against production v1 on `eval_set_v1`: REJECTED — pr_auc_macro improvement not significant at 95% CI (lower bound −0.0198 ≤ 0). On the frozen eval set: accuracy=0.5063, macro_f1=0.4710, per-class F1 P0=0.5560/P1=0.5318/P2=0.3252. So XGBoost on dense SVD-compressed text features does not beat the linear baseline on this dataset — consistent with the small (7,763-row) training set and the SVD compression discarding sparse n-gram signal the LogReg model uses directly; per `docs/candidate_models/overview.md` a dense LogReg control arm on the same SVD features (isolating "did SVD hurt" from "did the model family swap hurt") remains an open follow-up, not attempted here. `train_baseline.py`'s `MODEL_VERSION` constant is hardcoded `"baseline_v0"`, so its default (unpassed) `--model-out` path collides with production's own artifact file (`data/models/kubernetes-kubernetes/baseline_v0/model.joblib`) regardless of `--model-type`; ran with an explicit `--model-out data/models/kubernetes-kubernetes/xgboost_v1/model.joblib` to avoid overwriting production's real artifact — a CLI-flag choice, not a code change, matching the same pattern `baseline_v1_diagnostics` already uses in this tree. Code unmodified; production v1 unchanged. `uv run python -m signalscore.training.tune_xgboost --train data/processed/kubernetes-kubernetes/train.jsonl --val data/processed/kubernetes-kubernetes/val.jsonl --budget 60 --seed 42` then `uv run python -m signalscore.training.train_baseline --train data/processed/kubernetes-kubernetes/train.jsonl --val data/processed/kubernetes-kubernetes/val.jsonl --model-type xgboost --model-hyperparams '{"subsample": 0.6, "reg_lambda": 1.0, "reg_alpha": 1.0, "min_child_weight": 10, "max_depth": 6, "learning_rate": 0.01, "colsample_bytree": 0.6}' --model-out data/models/kubernetes-kubernetes/xgboost_v1/model.joblib --metrics-out data/models/kubernetes-kubernetes/xgboost_v1/metrics.json` then `uv run python -m signalscore.evaluation.run_gate --candidate-version 17 --notes "Experiment 3: XGBoost on dense SVD(n_components=100)-compressed TF-IDF + is_member_plus vs. production v1"` |
| 2026-09-02 | logreg_bge (Experiment 4a, candidate v32, run 946eb51653124168bf0b8ae048fd3367): bge-small-en-v1.5(384d, normalized, pinned revision 5c38ec7c) + is_member_plus -> LogisticRegression(C=1.0, penalty=l2) | kubernetes-kubernetes | 0.5023 | 0.5569 | 0.1982 | fail | Experiment 4a (`docs/candidate_models/overview.md`): TF-IDF replaced entirely by BGE sentence embeddings — first BGE candidate, LogReg only. Text field unchanged (title + first 2000 body chars, same input TF-IDF used), tokenizer-native 512-token truncation. On `--val`: accuracy=0.4852, macro_f1=0.4750, per-class F1 P0=0.4315/P1=0.4412/P2=0.5523, pr_auc_macro=0.4999, brier=0.1985. On the frozen eval set: accuracy=0.4899, macro_f1=0.4707, per-class F1 P0=0.5569/P1=0.4798/P2=0.3754. Stratified: CI-flake subset (n=370, 15.5% of eval set) macro_f1=0.4819 (P0=0.6738/P1=0.3274/P2=0.4444); non-CI subset (n=2010) macro_f1=0.4618 (P0=0.5155/P1=0.4981/P2=0.3716) — close to each other, so the aggregate result is not being carried by the CI-flake cluster. Gated for real via `run_gate.py` against production v1 on `eval_set_v1`: REJECTED — pr_auc_macro improvement not significant at 95% CI (lower bound −0.0129 ≤ 0). Note: production's own minority_f1 on the frozen eval set is 0.5569's counterpart at ~0.5509 (test split), not the 0.4824 in this file's first row — that earlier number was reported on `--val`, not the frozen test split; gate comparisons always use the frozen eval set, and the two splits are not directly comparable. First real gate run for this candidate hit a pre-check contract failure — `predict_proba took 4.525s on 200 rows, exceeding the 2.0s bound` — because `contract.py`'s latency check called `score_artifact()` without a warm BGE model, so it was timing a one-time cold model load, not per-request inference cost. Fixed in `contract.py` (warm-load the BGE model once before starting the clock, mirroring how the serving process actually behaves) and re-ran; the fix is a real, permanent correction landed in this branch, not a one-off workaround. Cross-repo transfer validation deferred to a future feature per direct instruction — this result is in-domain only. `uv run python -m signalscore.training.train_baseline --train data/processed/kubernetes-kubernetes/train.jsonl --val data/processed/kubernetes-kubernetes/val.jsonl --model-type logreg_bge --model-out data/models/kubernetes-kubernetes/logreg_bge_v1/model.joblib --metrics-out data/models/kubernetes-kubernetes/logreg_bge_v1/metrics.json` then `uv run python -m signalscore.evaluation.run_gate --candidate-version 32 --notes "Experiment 4a: BGE embeddings + LogReg vs. production v1"` |
| 2026-09-02 | xgboost_bge (Experiment 4b, candidate v33, run f105a065c5a14c6aaf434f8fabad3dac): bge-small-en-v1.5(384d, normalized, pinned revision 5c38ec7c) + is_member_plus -> XGBClassifier(defaults, no hyperparameter sweep) | kubernetes-kubernetes | 0.4706 | 0.5280 | 0.2002 | fail | Experiment 4b (`docs/candidate_models/overview.md`): second BGE candidate — same embeddings as 4a, XGBoost instead of LogReg, fed the dense embedding matrix directly (no SVD compression — BGE's 384 dims are already dense and small, unlike Experiment 3's sparse TF-IDF that needed compression). Un-litigating XGBoost here is deliberate: its one prior loss (Experiment 3) was on a *linear* SVD compression of sparse TF-IDF, a different feature geometry than BGE's non-linear semantic embedding — not a rerun of an answered question. No hyperparameter tuning attempted (unlike Experiment 3's `tune_xgboost.py` sweep) — trained with XGBoost defaults; a tuned arm remains an open follow-up if this feature source is revisited. On `--val`: accuracy=0.4875, macro_f1=0.4719, per-class F1 P0=0.4120/P1=0.4503/P2=0.5534, pr_auc_macro=0.4842, brier=0.1977. On the frozen eval set: accuracy=0.4845, macro_f1=0.4630, per-class F1 P0=0.5280/P1=0.5120/P2=0.3490. Stratified: CI-flake subset (n=370) macro_f1=0.4710 (P0=0.6636/P1=0.358/P2=0.3913); non-CI subset (n=2010) macro_f1=0.4512 (P0=0.4749/P1=0.5319/P2=0.3469) — production shows the same CI-flake-favors-P0 pattern (CI-flake macro_f1=0.4986, non-CI macro_f1=0.4733), so this isn't BGE-specific. Gated for real via `run_gate.py` against production v1 on `eval_set_v1`: REJECTED on two independent grounds — pr_auc_macro improvement not significant at 95% CI (lower bound −0.0500 ≤ 0), and minority_f1 regressed (candidate 0.5280 < production 0.5509 on the frozen eval set). Neither BGE candidate (4a or 4b) beat production in-domain; cross-repo transfer validation — BGE's actual justification — remains untested, deferred to a future feature per direct instruction. `uv run python -m signalscore.training.train_baseline --train data/processed/kubernetes-kubernetes/train.jsonl --val data/processed/kubernetes-kubernetes/val.jsonl --model-type xgboost_bge --model-out data/models/kubernetes-kubernetes/xgboost_bge_v1/model.joblib --metrics-out data/models/kubernetes-kubernetes/xgboost_bge_v1/metrics.json` then `uv run python -m signalscore.evaluation.run_gate --candidate-version 33 --notes "Experiment 4b: BGE embeddings + XGBoost vs. production v1"` |
