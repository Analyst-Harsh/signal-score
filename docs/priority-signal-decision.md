# Decision Memo — How We Extract the Priority Signal

**Status:** proposed · **Date:** 2026-08-25 · **Scope:** target definition (A) + input representation (B)
**Constraint added mid-analysis:** the shipped model must be **repo-generic**. K8s is the training
corpus only because it has the cleanest `priority/*` taxonomy at volume. `/score` must work on an
arbitrary repo whose labels, templates, SIG structure and conventions we have never seen.

Every number below comes from a script actually run over
`data/raw/kubernetes-kubernetes/*.jsonl` (12,025 rows / 11,991 unique issues). Anything I did not
measure is marked **not measured**. No estimates are dressed up as measurements.

---

## TL;DR recommendation

- **Ship title+body text as the only input.** Structured GitHub metadata is either leakage or
  k8s-specific; under the repo-generic constraint nearly all of it is banned. The serving contract
  is: **`title`, `body`, `author_association` (optional), `created_at` — nothing else.**
- **Reframe the target as ordinal 3-class, and gate on binary `critical-urgent vs not`.**
  Report macro-F1, but the promotion gate decides on **expected cost** under an explicit asymmetric
  cost matrix plus **recall on critical-urgent**. Missing a critical is not symmetric with
  over-flagging a backlog item.
- **v0 = TF-IDF word(1,2) + char(3,5) → linear model, time-split.** A stdlib-only multinomial NB on
  title+body already scores **52.3% acc / 46.8 macro-F1** on a strictly time-based split vs a
  **20.8%** majority-class baseline. That is real signal, from text alone, with zero metadata.
  TF-IDF + a calibrated linear model is the floor to beat, not the goal.
- **BGE embeddings must earn their slot** against that v0 on the **cross-repo holdout**, not on the
  in-domain set. Cross-repo transfer is where embeddings should win and TF-IDF should collapse
  (k8s vocabulary is not Rust vocabulary). That is the experiment; do not pre-assume the outcome.
- **The single biggest risk is not leakage — it is corpus specificity.**
  **54.4% of `priority/critical-urgent` issues carry `kind/flake` or `kind/failing-test`**, and
  **50.9%** have a CI-shaped title (`broken test run`, `[k8s.io]`, `e2e`, `kubemark`, `gke-`).
  Trained naively, the model learns *"is this a Kubernetes CI flake report?"* — a k8s-infra artifact
  that transfers to no other repo. See **The corpus-specificity trap** below; this drives the plan.

---

## Data reality check — measured

### Corpus shape

| Metric | Value |
|---|---|
| Rows in files / unique issue numbers | 12,025 / **11,991** (34 issues appear in 2 query files) |
| Pull requests mixed in | **0** — clean, all real issues |
| Class balance (by `_queried_label`) | important-soon **40.7%** (4,883) · backlog **40.6%** (4,871) · critical-urgent **18.7%** (2,237) |
| `_queried_label` still present in current `labels` | **100%** (0 drift-away) |
| Issues with >1 `priority/*` label | **92** (0.77%) |
| Other `priority/*` labels found in corpus | `priority/important-longterm` **56** · `priority/awaiting-more-evidence` **2** |
| State | closed **11,691** (97.5%) · open **300** |
| Author association | CONTRIBUTOR 5,270 · NONE 3,428 · MEMBER 3,293 · (no OWNER) |
| Issue authors that are bots (`user.type == Bot`) | **0** — all `User` |
| Distinct labels in corpus | 166 (70 appear ≥50×); **79.5%** are k8s-namespaced (`sig/`,`area/`,`kind/`,`triage/`,`lifecycle/`,`priority/`,`needs*`) |
| Issues whose only labels are `priority/*` | 682 |
| `sub_issues_summary` / `issue_dependencies_summary` non-zero | **0 / 0** — dead fields, ignore |

### Text quality

| Metric | Value |
|---|---|
| Body empty or whitespace-only | **114 (0.95%)** — 1 literal `null` |
| Body chars p10 / p50 / p90 / p99 / max | 152 / **707** / 3,387 / 20,891 / 245,789 |
| Body words p50 / p90 / p99 | **79** / 334 / 1,648 |
| Title chars p50 / p90 / max | 56 / 106 / 352 |
| Est. tokens (title+body, chars/4) p50/p75/p90/p95 | 196 / 418 / **863** / 1,389 |
| **Est. share exceeding BGE's 512-token window** | **19.9%** (2,390 issues) |
| Has fenced code block | **43.4%** |
| Has stack-trace-ish content (`panic:`, `goroutine N [`, `.go:N +0x`) | 2.8% |
| Prow slash-command in body (`/kind`, `/sig`, `/assign`…) | **16.4%** |
| HTML-comment (template boilerplate) share of all body chars | **0.7%** — negligible, do not over-engineer stripping |
| Template markers | `<!--` 783 · `**What happened**` 636 · `/kind bug` 534 · `### What happened` 459 |
| Non-English (>10% non-ASCII) / CJK present | 2 issues (0.02%) / 11 (0.09%) — **English-only corpus** |

### Temporal distribution and label drift (this is severe)

| Year | n | backlog | important-soon | critical-urgent |
|---|---|---|---|---|
| 2014 | 410 | 50% | 45% | 5% |
| 2015 | 3,824 | 54% | 38% | 8% |
| 2016 | 4,297 | 46% | 34% | 20% |
| 2017 | 685 | 16% | 49% | **35%** |
| 2018 | 566 | 10% | 51% | **39%** |
| 2019 | 672 | 14% | 56% | 30% |
| 2020 | 445 | 13% | 45% | **42%** |
| 2021 | 256 | 18% | 59% | 23% |
| 2022 | 176 | 26% | 57% | 16% |
| 2023 | 193 | 31% | 49% | 21% |
| 2024 | 188 | 41% | 38% | 20% |
| 2025 | 167 | 38% | 52% | 10% |
| 2026 | 112 | 35% | 49% | 16% |

**68% of the corpus is 2015–2016.** The class prior swings from 5%→42%→10% critical-urgent across
the decade. This is not noise; it is a changing labeling policy. A random split silently launders
that drift into the training set.

### Measured cost of a random split (stdlib multinomial NB, no deps installed)

| Split | Features | Acc | Macro-F1 | Recall (backlog / imp-soon / crit-urg) | Majority-class acc |
|---|---|---|---|---|---|
| **Time** (80/20, boundary 2018-08-20) | title only | 47.1 | 44.8 | 34.1 / 51.2 / 49.3 | **20.8** |
| **Time** | title+body | **52.3** | **46.8** | 19.3 / 57.7 / **66.7** | **20.8** |
| Random | title only | 53.9 | **52.3** | 59.9 / 50.4 / 48.6 | 40.9 |
| Random | title+body | 52.0 | 50.8 | 58.8 / 43.2 / 56.9 | 40.9 |
| Time, 2020+ only (n=1,537) | title+body | 49.4 | 45.1 | 76.4 / 33.8 / 36.6 | 51.0 |

Read this carefully. The random split looks *better* (52.3 vs 46.8 macro-F1) **and its baseline is
twice as high** (40.9% vs 20.8%) — because under a time split the test-period majority class is not
the training-period majority class. Random splitting would have reported a stronger model against a
weaker-looking baseline. That is exactly backwards. **Time split is non-negotiable.**

Also note: title-only beats title+body on backlog recall (34.1 vs 19.3) under the time split. Bodies
add signal for critical-urgent (+17.4pp recall) and cost it for backlog. That tension is real and is
a tuning knob, not a bug.

### The corpus-specificity trap (the headline risk)

| Signal | critical-urgent | important-soon | backlog |
|---|---|---|---|
| `kind/flake` or `kind/failing-test` label | **54.4%** | 22.4% | 22.1% |
| CI-shaped title (`broken test run`, `[k8s.io]`, `e2e`, `kubemark`, `gke-`, `flake`) | **50.9%** | 23.1% | 25.5% |
| `kind/failing-test` alone | 21.7% | 7.5% | **0.2%** |
| Keyword "flake" in text | 16.5% | 11.7% | 2.0% |
| k8s jargon in title (`kubelet`, `etcd`, `apiserver`, `pod`…) | 41.0% | 43.1% | 42.1% |

Over half of our positive class is *"a Kubernetes CI job broke."* The most discriminative single
feature in the corpus (`kind/failing-test`: 21.7% vs 0.2%) is a k8s test-infra convention. Top
repeated titles are literally `kubernetes-verify-master: broken test run` (18×),
`kubernetes-test-go: broken test run` (16×). Under the repo-generic constraint this is the dominant
threat to the whole project — far more than the obvious label leakage.

Note the last row: k8s jargon is **evenly distributed across classes** (41/43/42%). That is good
news — the domain vocabulary is not itself the priority signal, so it is dilution rather than
confound. The CI-flake cluster is the confound.

### Repo-agnostic severity lexicon coverage

| | critical-urgent | important-soon | backlog |
|---|---|---|---|
| Any of `data loss\|corrupt\|panic\|CVE\|regress\|crash\|deadlock\|security\|outage\|hang\|leak\|segfault\|OOM\|DoS` in title+body | **13.6%** | 13.3% | 7.4% |
| Same lexicon, title only | 4.5% | 3.9% | 1.6% |
| `data loss` | 0.0% | 0.2% | 0.0% |
| `CVE-` | 0.6% | 0.1% | 0.1% |
| `panic` | 4.2% | 2.9% | 1.3% |

**The severity lexicon is dead on arrival as a classifier.** It fires on 13.6% of criticals and
13.3% of important-soon — essentially no discrimination between the top two classes, and the
headline terms (`data loss`, `outage`, `CVE`) fire on <1%. Verdict below.

### Post-hoc metadata (all strongly predictive, all unusable — see Leakage)

| Field | critical-urgent | important-soon | backlog |
|---|---|---|---|
| Has milestone | 53.4% | 32.7% | 14.8% |
| Has assignee | 76.8% | 68.3% | 51.7% |
| `lifecycle/rotten` | 1.1% | 7.0% | **13.6%** |
| Mean comments | **26.3** | 13.2 | 9.8 |
| Median comments | 9 | 7 | 5 |
| Mean reactions | 0.39 | 0.82 | **1.51** |
| Closed | 100.0% | 98.1% | 95.8% |
| `closed_at` / `closed_by` present overall | 97.5% / 96.2% | | |
| `updated_at != created_at` | **100%** | | |

Note the reaction count is **inverted** (backlog gets ~4× the reactions of critical-urgent) —
community 👍 measures popularity, not urgency. That is precisely the proxy trap §1a of the design
doc warned about, and it is confirmed in our own data.

---

## Part A — Target definition

### Is `_queried_label` trustworthy ground truth?

**Yes, with three caveats — and it is unusually clean for a public-data label.**

- **Label integrity: excellent.** 100% of records still carry the label they were queried by. No
  label churn to reconcile. Only **0.77%** carry two `priority/*` labels — negligible; **drop them**
  (92 rows) rather than write reconciliation logic. Zero PRs contaminate the issue set. Zero bot
  authors. Prow enforces the taxonomy, which is why this corpus was chosen.
- **Excluded labels are correctly excluded.** `priority/awaiting-more-evidence` exists but appears
  only **2×** (it is a triage-pending state, not a priority — correctly out). `priority/failing-test`
  does **not** exist in this corpus (the k8s convention is `kind/failing-test`).
  `priority/important-longterm` appears **56×** — genuinely a fourth ordinal rung between
  important-soon and backlog. 56 rows is too few to train a class on. **Decision: exclude it** and
  document the exclusion; do not silently fold it into backlog.
- **Bot vs human labeling: not measured.** All *authors* are humans, but *who applied the priority
  label* requires the timeline/events API, which we did not fetch. Given `/priority` is a Prow
  slash-command typed by a SIG member, the label is human intent mediated by a bot. **Open question
  #1.** It matters only if a bot ever auto-applies priority in bulk.
- **The real caveat is time drift, not label noise.** The 5%→42%→16% swing in the critical-urgent
  prior across years means "critical-urgent" in 2015 and in 2025 are different thresholds applied by
  a different maintainer population. The label is trustworthy *conditional on its era*.

### Framing: ordinal, not nominal, not plain binary

**Recommendation: model as 3-class ordinal, gate on a binary derived decision.**

- **Nominal 3-class is wrong** — it treats confusing critical-urgent with backlog as no worse than
  confusing it with important-soon. Our own confusion structure says otherwise.
- **Pure binary (urgent-vs-not) throws away usable structure** and would produce a 18.7/81.3 split.
- **Ordinal is the honest structure** — backlog < important-soon < critical-urgent is a real ordering
  a maintainer would recognize. Practically: train a 3-class classifier with `class_weight="balanced"`
  and evaluate with an ordinal-aware cost matrix, or use two cumulative binary heads
  (`≥important-soon`, `=critical-urgent`). Start with the simpler 3-class + cost matrix; escalate to
  cumulative heads only if the confusion matrix shows off-by-two errors we cannot price away.
- The serving payload already promises `priority_score` (calibrated probability) + `priority_class`.
  Map: `priority_score = P(critical-urgent) + 0.5·P(important-soon)` — a monotone ordinal expectation,
  not an arbitrary blend. **This must be decided once and pinned**, since the dashboard's threshold
  tuning view depends on it.

### Class imbalance

18.7 / 40.7 / 40.6 is **mild**. The design doc's call — no resampling, handle at training time with
`class_weight="balanced"` — is correct; do not touch it. The *severe* imbalance problem here is
temporal, not class-count: the test period has a completely different prior from the training period.

### The repo-generic normalization layer (new constraint #3)

We need one canonical target and a documented mapping *into* it. Canonical scale:

| Canonical | Meaning |
|---|---|
| `P0_critical` | drop-everything; production-breaking, data-loss, security |
| `P1_soon` | must be addressed this cycle |
| `P2_backlog` | real but not scheduled |

Mapping table (training uses **only** the k8s row; every other row is **holdout-eval only**, per the
CLAUDE.md guardrail — this normalization does **not** relax it):

| Repo taxonomy | → P0 | → P1 | → P2 | Excluded |
|---|---|---|---|---|
| **k8s `priority/*`** (TRAIN) | `critical-urgent` | `important-soon` | `backlog` | `important-longterm` (56), `awaiting-more-evidence` (2) |
| **Rust `P-*`** (HOLDOUT) | `P-critical`/`P-high` | `P-medium` | `P-low` | — |
| **VS Code severity** (HOLDOUT) | `important` + `bug` | `feature-request` scheduled to milestone | `backlog`/`unreleased` | *mapping unverified — see Open question #3* |
| **Plain-English labels** (HOLDOUT) | `critical`,`urgent`,`blocker`,`sev1`,`security` | `high`,`important`,`bug` | `low`,`nice-to-have`,`someday`,`wontfix-ish` | ambiguous singletons |
| **No priority labels** (COLD START) | n/a — no ground truth exists | | | inference-only repo |

The mapping is a **static YAML file in the repo, hand-authored, reviewed, and versioned** — not
inferred, not embedding-matched. It is small, it is a value judgement, and a human must own it.
Rust's P-critical→P0 is defensible; VS Code's severity model does not cleanly ordinalize and I have
**not** verified it against real VS Code data (we have not downloaded it). Flag, do not guess.

---

## Part B — Input signal strategies

Ranking criteria, now re-weighted: **cross-repo transfer is first-class**, ahead of in-domain
accuracy. A representation that scores 5 points higher on k8s and collapses on Rust is worse than
one that scores 5 points lower and holds.

### B1. Title only

✅ Always available at t=0, on any repo, in any taxonomy. Zero leakage surface. Zero truncation
(**0 issues** exceed 512 tokens on title alone; p90 = 106 chars). Cheapest possible feature.
Measured **44.8 macro-F1** on the time split vs a 20.8% majority baseline — most of the achievable
signal, from one field.
✅ Under the time split it gives the **best backlog recall of any option tested (34.1% vs 19.3%)**.
❌ Loses 17.4pp of critical-urgent recall vs title+body (49.3 → 66.7).
❌ 50.9% of k8s criticals have machine-generated CI titles — the title channel is exactly where the
corpus-specificity trap bites hardest.
**Verdict: keep as an ablation arm and as the degraded-mode path for empty bodies (0.95% of issues).
Not the primary representation.**

### B2. Title + body (raw text)

✅ Best measured critical-urgent recall: **66.7%** on the time split. This is the class we care about.
✅ 99.05% of issues have a non-empty body; median 707 chars / 79 words — enough to model, short
enough to be cheap.
✅ **Fully repo-generic.** Every GitHub issue on earth has a title and a body. This is the only input
channel that survives the arbitrary-repo constraint intact.
❌ 43.4% contain code blocks; p99 body is 20,891 chars and the max is 245,789 — needs a truncation
policy (recommend: title + first 2,000 chars of body, which is what the measured baseline used).
❌ Costs backlog recall (34.1 → 19.3).
**Verdict: ✅ THE input channel. Everything else is a feature computed from it or banned.**

### B3. Structured GitHub metadata only

✅ Individually the strongest raw correlations in the dataset: milestone 53.4/32.7/14.8,
`kind/failing-test` 21.7/7.5/**0.2**, mean comments 26.3/13.2/9.8.
❌ **Every one of those is post-hoc.** Milestones, assignees, comments, lifecycle labels and
co-occurring `kind/*` labels are applied *by the same triage pass that applied the priority label*.
At t=0 they are all empty. Using them is training on the answer key.
❌ **And they are k8s-specific.** 79.5% of the label vocabulary is k8s-namespaced. `sig/*`, `area/*`,
k8s `kind/*`, k8s milestone naming — banned outright under constraint #1.
❌ Reactions are **inversely** correlated with priority (backlog 1.51 vs critical 0.39). A model that
found reactions would learn the wrong thing with confidence.
❌ `sub_issues_summary` and `issue_dependencies_summary` are **0 non-zero across all 11,991 rows** —
literally no signal.
**Verdict: ❌ BANNED, on two independent grounds (leakage AND repo-specificity). Only
`author_association` survives, and only as a 3-value generic enum — see whitelist.**

*On the proposed escape hatch — embedding co-occurring label TEXT to get a repo-agnostic label
feature:* clever, and it does dodge the fixed-vocabulary objection. It does **not** dodge the
leakage objection, which is fatal on its own: those labels do not exist when a new issue arrives.
Embedding `"kind/failing-test"` instead of one-hot-encoding it makes the leak portable, not absent.
**Rejected.**

### B4. Regex / keyword rules

✅ Perfectly interpretable, zero cost, zero training, trivially repo-generic in form.
❌ **The measurement kills it.** A 16-term repo-agnostic severity lexicon fires on 13.6% of criticals
and **13.3% of important-soon** — no discrimination where it matters. `data loss` 0.0%, `outage`
0.1%, `CVE` 0.6%. On title only it drops to 4.5% coverage.
❌ Repo-specific regex (`kubelet`, `etcd`) is banned by constraint #2 anyway — and would not have
helped: k8s jargon is evenly split across classes (41/43/42%).
**Verdict: ❌ Not a model. Ship the lexicon as ~15 binary features inside the feature union (they
cost nothing and are the interpretability story for the critical class), never as a rule engine.
Do not expect them to earn their keep.**

### B5. TF-IDF (word + char n-grams) → linear model / XGBoost

✅ Beats the baseline today, measured, with a strictly worse model than what we would ship: plain
multinomial NB hit **52.3% acc / 46.8 macro-F1 vs 20.8% majority** on the time split. TF-IDF +
calibrated LogisticRegression should beat that.
✅ **Directly interpretable** — per-class coefficients give triage engineers "this is critical
because of these words." No other option answers that as cleanly.
✅ Char n-grams handle the 43.4% of bodies containing code and identifiers, and the 2.8% with stack
traces, without a tokenizer decision.
✅ Trivial ops: model artifact in the low MB, sub-millisecond CPU inference, no GPU, no cold start,
reproducible from a pinned `scikit-learn` version.
❌ **Vocabulary-locked — this is the weakness under the new constraint.** A k8s-fit vocabulary will
transfer poorly to Rust or VS Code. Expect a large cross-repo drop. **Not measured** (holdout repos
are not downloaded yet) and it must be measured before anything else.
❌ Will happily learn the CI-flake cluster if we let it.
**Verdict: ✅ v0. It is the baseline that everything must beat, and it may well survive as
production if embeddings do not clear the bar.**

### B6. BGE sentence embeddings → classical head

✅ **The strongest theoretical case for cross-repo transfer** — semantic vectors carry "this is a
crash that loses user data" across vocabularies in a way TF-IDF cannot. Under the repo-generic
constraint this is the option's entire justification, and it is a good one.
✅ Design doc already commits to BGE reused from DocMind — shared infra, not a new pipeline.
❌ **19.9% of issues (2,390) exceed BGE's 512-token window** at a chars/4 estimate. One in five
issues gets silently truncated. The mitigation (title + first N chars) is fine but must be an
explicit, pinned, tested policy — not an accident of the encoder's default.
❌ **Opaque.** "Why is this critical?" gets answered with a nearest-neighbour, not a reason.
❌ **Versioning hazard, and it is real:** a BGE version bump silently changes the entire feature
space and invalidates every stored vector and the frozen eval set's features. Mitigation is
mandatory, not optional: pin the exact HF revision SHA (not the tag), log
`{model_name, revision_sha, max_seq_len, pooling, normalize}` as MLflow params on **every** run, and
make the model I/O contract test assert them. A mismatch fails the promotion gate.
❌ Ops cost: model download and cold start on serving; CPU inference is viable at our latency but
**p99 not measured**. Artifact size in MLflow **not measured**.
**Verdict: ⚠️ Earn it. Not in v0. Promote to v1 only on a measured cross-repo win — which is exactly
the test it should pass.**

### B7. Hybrid — embeddings + TF-IDF + metadata union

✅ Usually the best in-domain number.
❌ The metadata third is banned (B3). So this reduces to embeddings + TF-IDF + ~15 lexicon flags.
❌ Doubles the feature pipeline surface, the drift-monitoring surface, and the failure modes, for a
win that must be *measured* twice over (beat TF-IDF in-domain AND cross-repo).
**Verdict: ⚠️ v2 at the earliest. Only after B6 has independently beaten B5.**

### B8. LLM-as-labeler / LLM feature extraction

✅ Would likely be the most accurate and the most repo-generic thing in this memo. Intellectually
honest to say so.
❌ **Violates the project's stated identity** — this is a classical-ML portfolio project (CLAUDE.md
one-liner, design doc §5). Swapping in an LLM deletes the thing being demonstrated.
❌ Per-request latency and cost are 2–4 orders of magnitude above a linear model on TF-IDF; a
weekly-retrain + drift-monitoring loop over an API you do not control is not the loop described in §2.
❌ **As a labeler specifically it is worse than what we have** — we already possess 11,991
maintainer-assigned labels. Replacing human ground truth with model output would be a downgrade.
**Verdict: ❌ Out of scope. Documented as a deliberate rejection, not an oversight.**

### Summary comparison

| Option | Repo-generic | Leak-free | In-domain (measured) | Cross-repo transfer | Interpretable | Ops cost | Verdict |
|---|---|---|---|---|---|---|---|
| B1 title only | ✅ | ✅ | 44.8 F1 | likely medium | ✅ | trivial | ablation + fallback |
| B2 title+body | ✅ | ✅ | **46.8 F1** | likely medium | ✅ | trivial | **the input channel** |
| B3 metadata | ❌ | ❌ | n/a (leaks) | ❌ | ✅ | trivial | **BANNED** |
| B4 regex/lexicon | ✅ | ✅ | 13.6 vs 13.3% — none | ~ | ✅✅ | none | features only, not a model |
| B5 TF-IDF → linear | ✅ | ✅ | ≥46.8 F1 | ⚠️ vocab-locked | ✅✅ | low | **v0 — ship this** |
| B6 BGE → head | ✅ | ✅ | not measured | ✅ best case | ❌ | med (512-tok, pinning) | v1 — must earn it |
| B7 hybrid | ✅ | ✅ | not measured | ⚠️ | ~ | high | v2 at earliest |
| B8 LLM | ✅ | ✅ | not measured | ✅ | ~ | ❌ prohibitive | **out of scope** |

---

## Leakage: allowed vs banned

### The serving contract — what the caller actually has at t=0

A brand-new issue, seconds after `POST /issues`, from a repo we may never have seen:

```json
{ "title": "...", "body": "...", "author_association": "NONE|CONTRIBUTOR|MEMBER",
  "created_at": "2026-08-25T..." }
```

**That is the complete input.** Anything not in that object is not a feature. If a field cannot be
populated by a webhook payload on issue-opened for an arbitrary repo, it does not exist.

### ✅ ALLOWED at inference

| Feature | Source | Why safe |
|---|---|---|
| `title` raw text | issue | present at t=0, universal |
| `body` raw text (truncated: title + first 2,000 chars) | issue | 99.05% non-empty, universal |
| Title/body length, word count | derived | derived from the above |
| Has code block / stack trace / URL / numbered list | derived | 43.4% / 2.8% measured; format signal, not triage output |
| Severity-lexicon flags (~15 repo-agnostic terms) | derived | weak (13.6% vs 13.3%) but free and interpretable |
| Template-shape flags (has structured headings, has HTML comments) | derived | **generic shape only** — never k8s section names |
| `author_association` ∈ {NONE, CONTRIBUTOR, MEMBER/COLLABORATOR/OWNER} | issue | GitHub-native, repo-independent, known at t=0. Measured: MEMBER-or-above 33.1/32.0/20.3 — weak but honest |
| `created_at` | issue | **for time-splitting only** — never a model feature |

### ❌ BANNED — post-hoc (does not exist at t=0)

`labels` (the target lives here), all co-occurring triage labels (`kind/*`, `triage/*`,
`lifecycle/*`, `needs-triage`), `milestone`, `assignee`/`assignees`, `comments` count, `reactions`,
`state`, `state_reason`, `closed_at`, `closed_by`, `locked`, `active_lock_reason`, `updated_at`
(differs from `created_at` for **100%** of rows — a pure post-hoc timestamp),
`pinned_comment`, `performed_via_github_app`, `issue_field_values`, `sub_issues_summary`,
`issue_dependencies_summary` (both **0 non-zero** anyway).

### ❌ BANNED — repo-specific (would exist at t=0 but does not generalize)

`sig/*`, `area/*`, k8s `kind/*` vocabulary, k8s milestone naming (`v1.x`), Prow slash-commands in
body (**16.4%** of bodies contain one — this is a k8s bot convention and must be **stripped in
preprocessing**, not modeled), k8s issue-template section headers (`**What happened**` 636×,
`### What happened` 459× — strip the header, keep the prose), repo-specific identifiers as regex
features, `repository_url` / repo name in any form.

### ❌ BANNED — anti-features (correlate the wrong way)

`reactions.total_count` — measured **inverse** relationship (backlog 1.51 vs critical-urgent 0.39).
Any future proxy target derived from close-time, comment volume, or reaction count. The design doc
§1a called this out; our data confirms it.

### Train/serve skew control

One `build_features(title, body, author_association) -> vector` function, imported by `training`,
`evaluation`, `serving`, and `monitoring`. No separate "training-time" path. The feature-pipeline
unit tests (gate step 1) assert the training call and the serving call on the same input produce
byte-identical vectors. Every banned field is dropped **at load time**, in the collection→features
boundary, so it is not physically present in the training frame — a whitelist, not a blacklist.

### Cold-start repo (constraint #6)

A repo with zero historical labels. **Good news: it changes nothing.** The whitelist above was
derived from the "brand-new issue" case and already assumes no label history. A cold-start repo is
the *default* case, not a special one — which is the strongest argument that the whitelist is
correct. What it does mean:
- We can never calibrate a per-repo threshold from that repo's history. `/score` must ship a
  global calibration; the dashboard's threshold-tuning view is how a repo owner adapts it manually.
- `confidence_note` should read `"drift-flagged"` when the input embedding/TF-IDF profile is far
  from the training distribution — which for an unseen repo will often be true, and should be said
  out loud rather than hidden behind a confident number.

---

## Recommendation — the staged plan

### v0 — the baseline that must be beaten (ship first)

- Target: 3-class ordinal on the canonical scale. Drop the 92 multi-priority issues, drop the 56
  `important-longterm`, drop the 2 `awaiting-more-evidence` → **~11,841 usable rows**.
- Input: `title + body[:2000]`, Prow slash-commands and template headers stripped.
- Model: TF-IDF word(1,2) + char_wb(3,5) → LogisticRegression, `class_weight="balanced"`,
  `CalibratedClassifierCV`.
- Split: **time-based**, frozen eval = the most recent 20% by `created_at`, DVC-added and git-tagged
  `eval-set-v1` per design §1b. The frozen set is chronologically last, forever; future refetches
  extend the *training* pool only. **This is a change from design §1b's stratified hash split** —
  that split is random-within-class and would reproduce the 20.8%→40.9% baseline inflation measured
  above. Stratification and temporal validity are in direct conflict here; temporal wins.
- **Dumbest baselines that must be beaten, all three:** (1) majority class — **20.8% acc** on the
  time-split test; (2) the 15-term severity lexicon rule; (3) title-only NB — **44.8 macro-F1**.
- Acceptance: beat all three, with calibrated probabilities (Brier/ECE reported), and pass gate
  steps 1 and 2. Log the row in `EXPERIMENTS.md` whether it wins or loses.

### v0.5 — measure genericity before adding any complexity

Download the holdout repos (Rust, VS Code), apply the normalization YAML, and score the v0 model
zero-shot. **This is the number that decides v1.** No training on holdout data, ever.

### v1 — BGE, if and only if it earns it

Same target, same split, same contract. Replace TF-IDF with BGE embeddings (pinned revision SHA,
explicit 512-token truncation policy) → LogisticRegression/XGBoost head.
**Acceptance — all three, per the promotion gate:**
1. Passes gate steps 1 & 2 (feature unit tests, I/O contract incl. the pinned-embedding assertion
   and the latency bound).
2. Gate step 3: beats v0 on the frozen held-out set by a **real margin** — non-overlapping bootstrap
   CIs on expected cost, not a point-estimate win.
3. **New gate step 3b (cross-repo): its degradation on the holdout repos is no worse than v0's.**
   If BGE wins in-domain but transfers worse than TF-IDF, it is **rejected** — that inverts its
   entire justification. Losses logged, not discarded.

### v2 — hybrid union. Only after v1 has independently beaten v0. Same three criteria.

### Metric, stated once

Primary gate metric: **expected cost** under an explicit asymmetric matrix, because missing a
critical is not symmetric with over-flagging a backlog item. Proposed starting matrix (**a value
judgement requiring human sign-off — Open question #2**):

| true \ pred | P0 | P1 | P2 |
|---|---|---|---|
| **P0** | 0 | 5 | **10** |
| **P1** | 1 | 0 | 3 |
| **P2** | 1 | 0.5 | 0 |

Reported alongside, never as the gate: macro-F1, per-class recall (critical-urgent recall is the
one a human will actually look at), PR-AUC on the derived binary, Brier/ECE.

### Cross-repo degradation bound (constraint #5)

Concrete criterion, checked at every promotion:

> Let `M_k8s` = macro-F1 on the frozen k8s eval set and `M_holdout` = macro-F1 on the normalized
> holdout repos. A candidate may not be promoted if
> **(a)** `M_holdout` falls below the holdout majority-class baseline + 5pp, **or**
> **(b)** `M_holdout / M_k8s` is more than **10pp lower** than the current production model's ratio.

(b) is the important half: it bounds *relative* degradation, so a candidate cannot buy in-domain
accuracy by over-fitting k8s conventions. The absolute threshold in (a) is a placeholder until
v0.5 gives us a real reading — **do not treat 5pp as measured.**

### Anti-CI-flake control (mandatory, given the headline risk)

Because **54.4%** of criticals are `kind/flake`/`kind/failing-test`, report a **stratified eval**:
macro-F1 on the CI-flake subset and on the non-CI subset, separately, at every promotion. A model
whose overall number is carried by the flake cluster is not a priority model; it is a CI-log
detector. If the non-CI slice is near baseline, that fact goes in `EXPERIMENTS.md` in bold. Consider
down-weighting or capping the flake cluster in training — **but measure the trade-off first**; do
not delete 20% of the corpus on a hunch.

---

## What we are explicitly NOT doing, and why

- **Not using any co-occurring label, milestone, assignee, comment or reaction feature.** Post-hoc
  (does not exist at t=0) *and* repo-specific. Two independent disqualifications.
- **Not embedding label text to make labels "repo-agnostic."** Fixes the vocabulary problem, leaves
  the leakage problem untouched. A portable leak is still a leak.
- **Not using a random or stratified-hash split.** Measured: it inflates the baseline from 20.8% to
  40.9% and would have flattered the model against a weaker yardstick. This overrides design §1b.
- **Not using reaction counts.** Measured inverse correlation; it would learn popularity, not urgency.
- **Not building a regex rule engine.** 13.6% vs 13.3% coverage — the lexicon does not separate the
  top two classes. Kept as ~15 free features for interpretability only.
- **Not starting with BGE.** Unearned complexity, 19.9% truncation, opaque, version-fragile. It gets
  one clean shot at beating a measured baseline on the criterion that actually matters (transfer).
- **Not using an LLM** anywhere in the scoring path or the labeling path. Deliberate; see B8.
- **Not training on `important-longterm` (56) or `awaiting-more-evidence` (2).** Too few; folding
  them into a neighbouring class would fabricate labels.
- **Not writing multi-priority reconciliation logic** for 92 rows (0.77%). Drop them.
- **Not stripping HTML comments aggressively.** They are 0.7% of body characters. Not worth code.
- **Not touching `sub_issues_summary` / `issue_dependencies_summary`.** Zero non-zero values.

## Open questions for the human

1. **Who applies the priority label — human or bot?** All 11,991 *authors* are human, but the
   labeling actor needs the timeline API. Not measured. Only matters if bulk auto-labeling exists.
2. **Sign off on the cost matrix.** The 10:1 P0-missed-as-P2 ratio is my proposal, not a measurement.
   It determines every promotion decision, so it needs a human owner.
3. **Is the VS Code severity mapping real?** I mapped it from memory of their label scheme without
   the data in hand. Verify against a real pull before it becomes a gate input. Rust's `P-*` is
   straightforward; VS Code's is not clearly ordinal.
4. **Should the 2014–2016 bulk (68% of the corpus) be down-weighted?** It is where the label
   convention is most different from today (5–8% critical vs ~16% now). Options: train on all with
   recency weighting, or restrict to 2017+ (n≈3,571). The 2020+-only run scored 45.1 macro-F1 on
   n=308 — too small to conclude anything. Needs an experiment, not an opinion.
5. **`priority_score` formula:** confirm `P(P0) + 0.5·P(P1)`. It is a monotone ordinal expectation,
   but the dashboard threshold view and any downstream consumer are pinned to whatever we choose.
6. **Do we accept losing backlog recall?** Title+body raises critical-urgent recall to 66.7% and
   drops backlog to 19.3%. Under the proposed cost matrix that is the right trade — but it should be
   an explicit decision, not a side effect.
7. **Volume supplement:** at ~11,841 usable rows we are not obviously data-starved, but the recent
   era is thin (2020+ is only 1,537 issues). Should we pull `kubernetes-sigs/*` now to thicken the
   modern regime, per design §1a?

---

*All figures produced by scratch scripts run against `data/raw/kubernetes-kubernetes/*.jsonl` on
2026-08-25. Scripts are in the session scratchpad, deliberately not committed — they are
measurement, not pipeline code. Every claim not backed by a run is marked "not measured."*
