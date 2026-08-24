# SignalScore — Final Design (Weeks 12–15)

**Scope note:** This design is deliberately self-contained. TriageBot integration and the
priority-queue dispatch refactor are **not** part of this build — see "Deferred Decisions"
at the bottom.

---

## 1. Problem Statement

A classical ML scoring service that predicts **issue priority**, trained initially on
public GitHub data, with the pipeline designed so TriageBot's own logs can become a richer
training signal later.

**Prediction target — locked: issue-priority**, not PR-merge-likelihood. Rationale:
issue-priority has a real downstream consumer (the future priority-queue mechanism, and
Risk-check routing once TriageBot integration happens); merge-likelihood doesn't map to
anything in the current architecture — TriageBot generates patches for human approval, it
doesn't seek external PR merges.

## 1a. Data Source Strategy — locked

Public GitHub issues don't self-label priority the way PRs self-label merged/unmerged, and
inventing a proxy label (e.g. "closed fast = urgent," "many comments = urgent") risks the
same leakage/confound trap caught during ML Fundamentals — proxies often measure something
else (controversy, not urgency) or leak post-hoc information.

**Fix: use a repo with an explicit, maintainer-assigned priority taxonomy as ground truth.**

- **Primary training source:** `kubernetes/kubernetes` issues, using its
  `priority/critical-urgent`, `priority/important-soon`, `priority/backlog` labels.
- **Volume/balance supplement (if needed):** other `kubernetes-sigs/*` repos that share the
  *same* Prow-bot-enforced `priority/*` taxonomy — safe to blend because the labeling
  convention and enforcement are identical, not just similar.
- **Do NOT blend repos with different/unrelated taxonomies** into training data (e.g. Rust's
  `P-high/P-medium/P-low`, VS Code's severity labels) — different maintainer cultures and
  thresholds mean the model would partly learn "which repo is this" instead of "how urgent
  is this."
- **Generalization test set (never trained on):** hold out a different-taxonomy repo (Rust
  or VS Code) purely to measure zero-shot cross-taxonomy generalization. This becomes a
  documented `EXPERIMENTS.md` number — "trained on K8s-convention labels, generalizes to
  Rust issues with X% degradation" — turning the multi-repo question into an honest
  experiment rather than a hidden confound.

---

## 2. Architecture (closed loop, no external dependencies)

```
Public GitHub Issues (bootstrap dataset)
        │
        ▼
Feature Pipeline
  • Text signals — length, stack traces, urgency language
  • GitHub metadata — author association, labels, comments, issue age
  • BGE embeddings — reused from DocMind (shared infra, not a siloed pipeline)
        │
        ▼
Train / Val / Test split
        │
        ▼
Multiple model configs (scikit-learn / XGBoost) — tracked in W&B
        │
        ▼
Best candidate registered in MLflow, tagged "staging"
        │
        ▼
┌─────────────── PROMOTION GATE ───────────────┐
│ 1. Feature-pipeline unit tests                │
│ 2. Model I/O contract test                    │
│    (schema, valid probabilities, no NaNs,     │
│    latency bound)                              │
│ 3. Head-to-head eval vs current production     │
│    on a FROZEN held-out set                    │
│    — metric: PR-AUC / minority-class F1        │
│      + calibration (Brier/ECE)                 │
│    — must beat production by a real margin     │
│      (not any point-estimate win — check CI    │
│      non-overlap or a minimum delta)            │
└────────────────────────────────────────────────┘
        │
   ┌────┴────┐
   │         │
 PASS       FAIL
   │         │
   ▼         ▼
tag →      stays "staging" / marked "rejected"
"production"   (never deleted — retrain history
   │            logs the loss as a real row)
   ▼
FastAPI /score endpoint (loads production-tagged model)
   │
   ▼
Response payload:
{
  "priority_score": 0.83,          // calibrated probability
  "priority_class": "high",
  "model_version": "v3-2026-08-20",
  "confidence_note": "in-distribution"  // or "drift-flagged"
}
   │
   ├──► Evidently AI — input drift + prediction drift on live traffic
   │         │
   │         ▼
   │    Weekly GitHub Actions retrain job
   │    (or out-of-band trigger on heavy drift)
   │         │
   │         └──► loops back to Feature Pipeline with fresh data
   │
   └──► Next.js Dashboard
             • Drift charts
             • Prediction explorer
             • Retraining history (wins AND losses)
             • Threshold tuning view
```

**Rollback path:** promoting a bad model is a one-line MLflow tag revert to the prior
production version — this should be explicitly tested once, not just documented.

---

## 3. Promotion Gate — Detailed Rules

1. **Every candidate starts quarantined** — registered as `staging`, never auto-promoted.
2. **Pre-checks gate the comparison, not just the promotion** — a candidate that fails unit
   tests or the I/O contract test never reaches the eval step.
3. **Comparison is always against current production**, on the same frozen held-out set —
   never re-shuffled between retrain cycles.
4. **Margin requirement, not "any win"** — a single point-estimate improvement can be noise
   on a modest held-out set. Require either a minimum delta or non-overlapping confidence
   intervals before promoting.
5. **Losses are logged, not hidden.** Retraining history shows every attempt — "candidate
   scored X vs production's Y, not promoted" — consistent with treating null results as
   first-class outcomes (same instinct as the ML Fundamentals calibration/importance work).
6. **Escalation is pattern-based, not single-failure-based.** One losing retrain is
   expected. Production surviving N consecutive retrain attempts *while drift is climbing*
   is the signal worth surfacing on the dashboard.

---

## 4. Feature Set (input side)

Since training starts on public data before TriageBot has its own logs:

| Category | Features |
|---|---|
| Text-derived | title/body length, stack-trace presence, error-text presence, urgency language |
| GitHub metadata | author association (OWNER/MEMBER/CONTRIBUTOR/NONE), label history, comment/reaction count, time-since-open, issue template used |
| Embeddings | BGE-large (reused from DocMind rather than a separate TF-IDF pipeline) |

---

## 5. Tech Stack

| Category | Tool |
|---|---|
| Modeling | scikit-learn / XGBoost |
| Experiment tracking | Weights & Biases |
| Model registry | MLflow (staging/production tags) |
| Serving | FastAPI |
| Drift monitoring | Evidently AI |
| Retraining automation | GitHub Actions (scheduled + drift-triggered) |
| Frontend | Next.js / TypeScript |
| Deployment | Fly.io / Railway |

---

## 6. Testing Requirements

- Feature-pipeline unit tests
- Model I/O contract test (schema + sanity bounds) — runs *before* a model is eligible to serve
- Rollback path — explicitly tested at least once, not just documented

---

## 7. Portfolio Differentiator

Not the model's raw accuracy — the **loop**: explicit metric-tradeoff reasoning (precision/
recall/calibration on an imbalanced problem) + a promotion gate that can say no + drift-
gated retraining + an honestly-logged history of both wins and losses.

---

## 8. Deferred Decisions (explicitly out of scope for Weeks 12–15)

- **TriageBot integration** (Researcher node calling `/score`, Risk-check routing using
  `priority_score`) — to be planned in a future session.
- **Priority-queue + worker-pool dispatch refactor** for TriageBot (tiered queues, weighted
  round-robin to avoid starvation, SQS + ECS Fargate autoscaling workers) — deferred to
  **Capstone platform integration (Weeks 20–24)**, alongside the Terraform/AWS infra work.
  This replaces TriageBot's current sequential run loop.

See `signalscore-architecture.mermaid` for the visual diagram of the scope above.
