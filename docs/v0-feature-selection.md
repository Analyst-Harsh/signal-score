# v0 Feature Selection — Skeptical Second Pass

**Status:** proposed · **Date:** 2026-08-25 · **Reviews:** `docs/priority-signal-decision.md`
**Scope:** which features `build_features()` actually computes for v0.

Every number below was re-derived by scripts run against
`data/raw/kubernetes-kubernetes/*.jsonl` in this session. Scripts are in the session scratchpad,
deliberately not committed. Anything not measured is marked **not measured**. Nothing is estimated.

`.venv` now has the `ml` extra installed (`uv sync --extra ml`) — sklearn 1.9.0 / xgboost 3.4.1 —
because the prior pass could only run a stdlib NB and this review needed real ablations.

---

## TL;DR — the final v0 feature list

**One text channel, no derived numeric block. This is a reduction from the drafted list, not an
addition.**

1. `text = strip(title) + "\n" + strip(body)[:2000]` — HTML comments removed, Prow slash-command
   lines removed, template section headers removed (keep the prose under them).
2. **Digit normalization on `text`, mandatory:** every digit → `#`, hex runs ≥7 chars → `HASH`,
   month names → dropped. See *Leakage re-audit* — the raw model's top-30 P0 word features include
   `sep`, `06`, `08`, `2017`, `11e7`, `v10`. Those are era and build-id tokens, and under a time
   split they are pure train-period artifacts.
3. TF-IDF **word (1,2)**, `min_df=3`, `sublinear_tf=True`, `strip_accents="unicode"` on `text`.
4. TF-IDF **char_wb (3,5)**, `min_df=5`, `max_features=300_000`, `sublinear_tf=True` on `text`.
5. `author_association` → single binary `is_member_plus` ∈ {MEMBER, OWNER, COLLABORATOR}.
   **Keep with an explicit caveat:** it is the only derived feature that was never measurably
   harmful, and it is the only non-text field in the serving contract. Its measured contribution is
   +0.85 macro-F1 across 3 time folds and −0.22 on the 80/20 split — i.e. indistinguishable from
   zero. It is kept because it costs one bit and one line, not because it was shown to help.
6. **Computed and logged, but NOT fed to the model:** `title_chars`, `body_chars`, `body_words`,
   `has_code_block`, `has_stack_trace`, `has_url`, `has_logline`, `has_excl`, `code_ratio`,
   `any_lex`, `is_ci_flake_shaped`. These are the **drift-monitoring vector** for Evidently and the
   **stratification keys** for the mandated CI-flake-vs-non-CI eval. They are cheap, interpretable,
   and genuinely useful — just not as model inputs. Ship them in the feature frame, exclude them
   from the model matrix, and say so in one comment.

**Everything else on the drafted list is cut.** Details and numbers below.

**Single most important change vs the draft:** the ~15 severity-lexicon flags, the length features,
and the shape flags are **removed from the model** and demoted to monitoring. Adding all drafted
derived features to TF-IDF measured **−2.75 macro-F1** on the 80/20 time split with a bootstrap 95%
CI of **[−4.38, −1.10]** — a statistically significant *loss*, not a free win. "Free and
interpretable" was a rationalization; the features are not free, they cost accuracy.

**Second most important change:** digit normalization is now mandatory preprocessing. It was not in
the draft, and without it the model demonstrably keys on calendar-year tokens under a time split.

---

## Verification of the prior pass's numbers

| Claim in `priority-signal-decision.md` | Re-measured | Match |
|---|---|---|
| 12,025 rows / 11,991 unique issues | 12,025 / 11,991 | ✅ |
| 92 issues with >1 `priority/*` label | 92 | ✅ |
| `important-longterm` 56 / `awaiting-more-evidence` 2 | 56 / 2 | ✅ |
| **Usable rows ≈ 11,841** | **11,899** | ❌ **see below** |
| Class balance 18.7 / 40.7 / 40.6 | 18.5 / 40.8 / 40.7 (P0 2,213 · P1 4,883 · P2 4,861) | ✅ |
| `kind/flake` or `kind/failing-test` 54.4 / 22.4 / 22.1 | 54.50 / 22.40 / 22.07 | ✅ |
| `kind/failing-test` alone 21.7 / 7.5 / 0.2 | 21.51 / 7.47 / 0.25 | ✅ |
| CI-shaped **title** 50.9 / 23.1 / 25.5 | **41.89 / 19.35 / 25.20** | ⚠️ partial |
| Severity lexicon any-term 13.6 / 13.3 / 7.4 | **16.04 / 14.75 / 8.66** | ⚠️ close, direction holds |
| `panic` 4.2 / 2.9 / 1.3 | 4.34 / 3.13 / 1.36 | ✅ |
| `CVE-` 0.6 / 0.1 / 0.1 | 0.59 / 0.14 / 0.08 | ✅ |
| `data loss` 0.0 / 0.2 / 0.0 | 0.05 / 0.18 / 0.04 | ✅ |
| has fenced code block 43.4% | 43.9% overall (54.99 / 41.39 / 40.12 by class) | ✅ |
| stack-trace-ish 2.8% | 2.6% overall (4.07 / 3.03 / 1.28 by class) | ✅ |
| `author_association` MEMBER-or-above 33.1 / 32.0 / 20.3 | **33.08 / 31.99 / 20.35** | ✅ exact |
| Author association counts CONTRIBUTOR/NONE/MEMBER, no OWNER | CONTRIBUTOR 5,252 · NONE 3,422 · MEMBER 3,283 (usable set) | ✅ |

### The one real error: usable row count

The memo computes `11,991 − 92 − 56 − 2 = 11,841`. That **double-subtracts**. Measured
decomposition of the 92 multi-priority issues:

- 34 carry two of the core three (`critical-urgent`/`important-soon`/`backlog`) — genuinely ambiguous.
- 56 are `important-longterm` paired with a core label.
- 2 are `awaiting-more-evidence` paired with a core label.
- 34 + 56 + 2 = 92. **There are zero `important-longterm`-only or `awaiting-more-evidence`-only
  issues in this corpus** — those labels were never queried as their own file, so they only ever
  appear as a second label on an already-collected issue.

So: **drop all 92 → 11,899 usable rows.** (Keeping the 58 non-core-conflict rows under their core
label would give 11,957; not recommended — dropping all 92 is the simpler rule and the difference is
0.5% of the corpus.) All numbers in this memo use the 11,957-row frame from the analysis script
except the ablations, which are unaffected by the 58-row difference. **Fix the count in the prior
memo.**

### The two partial mismatches

- **CI-shaped title 50.9% → 41.89%.** I used the memo's own stated regex
  (`broken test run|[k8s.io]|e2e|kubemark|gke-|flake`, case-insensitive, title only). I could not
  reproduce 50.9% on titles. The direction and the qualitative conclusion (CI shape is ~2× enriched
  in P0) hold, and the *label*-based measure (54.5%) reproduces exactly — which is the number the
  memo's headline risk actually rests on. **The headline risk stands; the title number should be
  corrected to 41.9%.**
- **Lexicon 13.6% → 16.04%.** My regex set differs slightly (I include `vulnerab`, `OOMKill`,
  `SIGSEGV`, word-boundary variants). Higher coverage, same verdict: 16.04 vs 14.75 between P0 and
  P1 is still no discrimination where it matters.

---

## The evidence that decides everything: real ablations

Not available to the prior pass (no sklearn). Setup: 11,957 rows sorted by `created_at`,
TF-IDF word(1,2) + char_wb(3,5) → `LogisticRegression(class_weight="balanced")`. Derived features
appended as a dense block, min-max scaled (standard-scaled was also run — see note).

### 80/20 time split (boundary 2018-07-30, test n=2,392, majority baseline 51.5% acc)

| Variant | acc | macro-F1 | recall P0/P1/P2 | non-CI F1 | CI F1 |
|---|---|---|---|---|---|
| **A. TF-IDF only** | 53.5 | **50.96** | 63.0 / 56.0 / 34.6 | 45.51 | 37.45 |
| B. + severity lexicon | 52.2 | 49.59 | 63.3 / 53.9 / 33.2 | 45.30 | 39.24 |
| C. + author_association | 53.6 | 50.71 | 63.8 / 56.5 / 32.8 | 45.91 | 37.16 |
| D. + length features | 53.3 | 50.80 | 63.0 / 55.7 / 34.6 | 45.00 | 37.94 |
| E. + shape flags | 53.1 | 50.50 | 64.4 / 54.8 / 33.8 | 45.11 | 37.46 |
| F. + my new candidates | 50.7 | 47.88 | 69.5 / 48.9 / 30.0 | 43.00 | 37.59 |
| G. + all drafted | 52.8 | 50.23 | 67.6 / 52.8 / 33.0 | 47.11 | 36.92 |
| H. + everything | 50.9 | 48.44 | 73.0 / 46.8 / 31.4 | 44.43 | 37.20 |
| J. derived only, no TF-IDF | 36.9 | 35.62 | 61.1 / 26.3 / 30.8 | 37.10 | 26.06 |
| K. lexicon only, no TF-IDF | 23.2 | 21.30 | 16.1 / 7.5 / 71.4 | 29.91 | 7.92 |

**Bootstrap (1,000 resamples) on the same test set:**
- `H − A` = **−2.75 macro-F1, 95% CI [−4.38, −1.10], P(delta > 0) = 0.00.** Adding the full derived
  block is a significant loss.
- `C − A` = −0.22, 95% CI [−1.20, +0.64], P(delta > 0) = 0.32. Author association is noise.

### Because one split is not evidence: 3 forward-chaining time folds

Train on everything before the fold, test on the fold. Mean macro-F1, delta vs TF-IDF-only:

| Variant | fold F1s (std-scaled) | mean | Δ | mean (min-max scaled) | Δ |
|---|---|---|---|---|---|
| A. TF-IDF only | 32.98 / 53.40 / 49.12 | 45.17 | — | 45.17 | — |
| B. + lexicon | 33.47 / 51.72 / 49.22 | 44.80 | −0.37 | 45.24 | +0.07 |
| C. + author_assoc | 33.59 / 53.95 / 50.12 | 45.89 | **+0.72** | 46.02 | **+0.85** |
| D. + lengths | 32.99 / 53.04 / 48.53 | 44.85 | −0.32 | 45.23 | +0.06 |
| E. + shape flags | 33.17 / 53.28 / 48.27 | 44.91 | −0.26 | 44.76 | −0.41 |
| F. + new candidates | 34.02 / 53.66 / 47.18 | 44.95 | −0.22 | 45.05 | −0.12 |
| G. + all drafted | 34.35 / 52.96 / 49.36 | 45.56 | +0.39 | 45.97 | +0.80 |
| H. + everything | 34.21 / 54.35 / 49.50 | 46.02 | +0.85 | 46.25 | +1.08 |
| Z. + has_excl, has_logline only | 33.18 / 53.23 / 48.58 | 45.00 | −0.17 | 44.99 | −0.18 |

**Read these two tables together, honestly.** Fold-to-fold variance is **±10 macro-F1** (32.98 to
53.40 for the *same* model). Every derived-feature delta is within ±1.1. The 80/20 bootstrap says
`H` is significantly worse; the 3-fold mean says `H` is +1.08. **The correct conclusion is not
"derived features help" or "derived features hurt" — it is "derived features are not measurably
distinguishable from noise, and at least one careful measurement says they actively hurt."**
Given that, the lazy and correct call is to not ship them in the model. A feature that cannot be
shown to help is a feature that costs pipeline surface, drift-monitoring surface, and train/serve
skew risk for nothing.

**A methodology note that mattered:** the first pass of this ablation standard-scaled the dense
block, which gave it magnitudes far above the L2-normalized TF-IDF rows and let it eat the
regularization budget. That alone flipped several deltas by ~0.4 F1. Min-max scaling is the fairer
comparison and is what the tables above report as the second column. **I nearly condemned these
features on a scaling artifact.** Anyone re-running this must scale the dense block onto TF-IDF's
range or the ablation is invalid.

---

## Feature-by-feature verdict

| # | Drafted feature | Verdict | Number behind the call |
|---|---|---|---|
| 1 | TF-IDF word(1,2) on title+body[:2000] | **KEEP** (+ digit normalization) | 50.96 macro-F1 vs 51.5% majority-acc baseline; the entire model |
| 2 | TF-IDF char(3,5) | **KEEP** | not separately ablated — **not measured** as a standalone contribution. Kept on the prior pass's reasoning (43.9% of bodies contain code; identifiers and stack frames tokenize badly). Flag: this deserves its own ablation in the v0 run. |
| 3 | title/body length, log-scaled | **CUT from model, keep for monitoring** | +0.06 / −0.32 macro-F1 across folds. Also a leakage suspect — see re-audit |
| 4 | title/body word count | **CUT** | redundant with #3 (Pearson not computed, but char and word counts of the same field are near-collinear by construction); same null ablation |
| 5 | `has_code_block` | **CUT from model, keep for monitoring** | 54.99 / 41.39 / 40.12 — separates P0 but not P1 from P2; block E = −0.41 |
| 6 | `has_stack_trace` | **CUT from model, keep for monitoring** | 4.07 / 3.03 / 1.28 — fires on 2.6% of the corpus. 311 issues total. Cannot move a macro-F1 |
| 7 | `has_url` | **CUT** | 78.58 / 58.32 / 52.95 — the best-separating single shape flag, and still inside a block measuring −0.41. Strongly suspect it is a CI-flake proxy (flake reports paste Prow links) |
| 8 | `has_numbered_list` / `has_bullet_list` | **CUT** | 3.84 / 7.04 / 7.12 and 22.23 / 24.68 / 15.63 — numbered list is *inversely* related to P0 |
| 9 | ~15 severity-lexicon flags | **CUT** | See below — the strongest cut in this memo |
| 10 | `author_association` 3-value ordinal | **KEEP as 1 binary, not 3 values** | 33.08 / 31.99 / 20.35; contribution +0.85 / −0.22 = noise. Binary `is_member_plus` captures the only real split (the P2 gap); the NONE-vs-CONTRIBUTOR distinction is 15.78 vs 19.05 P0-rate — nothing |
| — | `created_at` as split key only, never a feature | **KEEP the rule, and extend it to text** | See digit-normalization finding |

### #9, the severity lexicon — cut, and the "free + interpretable" argument is wrong on both counts

Three independent reasons, each sufficient:

1. **It does not discriminate where it matters.** Any-term coverage 16.04 (P0) vs 14.75 (P1) vs 8.66
   (P2). The P0−P1 gap is **1.3 percentage points**. Individually, half the terms point the *wrong*
   way: `crash` 1.99 / 2.29 / 1.13, `hang` 0.59 / 0.92 / 0.78, `OOM` 0.77 / 1.11 / 0.66,
   `corrupt` 0.23 / 0.41 / 0.08, `data loss` 0.05 / 0.18 / 0.04, `outage` 0.09 / 0.25 / 0.12.
2. **It is not free — it measurably costs.** Variant B is −1.37 macro-F1 on the 80/20 split. Variant
   K (lexicon alone) scores **21.30 macro-F1** and predicts P2 for 71.4% of the test set — a
   rule engine on this lexicon is worse than guessing the majority class.
3. **It is fully redundant with TF-IDF.** I checked the fitted word vocabulary directly:
   **21 of 22 lexicon terms are already in it** (`panic`, `crash`, `crashes`, `regression`,
   `deadlock`, `cve`, `data loss`, `corrupt`, `security`, `outage`, `hang`, `hangs`, `leak`, `oom`,
   `dos`, `broken`, `goroutine`, `traceback`, …). The single miss is `segfault`, which appears in
   **18 issues** and is covered by char 3-5-grams anyway. A logistic regression on TF-IDF already
   has a free coefficient on every one of these tokens, fit on the actual data rather than on a
   hand-authored prior. The lexicon flag is a *duplicate column with a worse estimator behind it*.

**On the interpretability argument specifically:** the interpretability story is already fully
served by the TF-IDF coefficients — they are per-token and per-class, and I printed them below. The
lexicon adds no explanatory power that `coef_["panic"]` does not already provide, and it adds a
maintained regex list, 15 columns of train/serve skew surface, and a monitoring obligation. That is
scope creep wearing an interpretability costume.

**Partial-value check, as asked:** does the lexicon separate P2 from the top two even if it can't
split P0/P1? Yes, weakly: 16.04 / 14.75 vs **8.66** — a 6.7pp gap. And on the non-CI subset it looks
genuinely monotone: **24.13 / 15.99 / 9.50**. That is the strongest case anyone can make for it.
It still does not survive: on the **2019+ non-CI slice (n=1,372)** — i.e. the regime the time-split
test set actually lives in — that ordering collapses (`any_lex` 24.22 / 23.17 / **27.04**, inverted).
The separation is a property of the 2015–2016 bulk, not of the era we deploy into. That is exactly
why it shows up in class-conditional tables and vanishes in a time-split ablation.
**Verdict: cut from the model. Keep `any_lex` as one monitoring scalar.**

### #10, author_association — keep, but as one bit and with the caveat stated

It is the only feature with a consistently positive sign across all 3 folds (+0.72 / +0.85), it is
in the serving contract, and it costs one line. But the 80/20 bootstrap CI is [−1.20, +0.64] and
straddles zero — this is **not** a demonstrated win. The measured structure is a P2 effect, not a
P0 effect: MEMBER-or-above rate is 33.08 / 31.99 / **20.35**, i.e. it distinguishes "backlog" from
"not backlog" and says nothing about P0-vs-P1. Encoding it as a 3-value ordinal is unjustified:
row-normalized, NONE authors are 19.05% P0 and CONTRIBUTOR authors are 15.78% P0 — the "ordinal"
ordering is not even monotone in P0. **One binary. Not three levels.**

---

## New candidates I considered, measured, and mostly rejected

All measured on the full 11,957-row frame (rates in %, P0 / P1 / P2), and again on 2019+ where the
era effect matters.

| Candidate | Full corpus | 2019+ | Call |
|---|---|---|---|
| `has_excl` (any `!` in title or body) | **25.44 / 15.05 / 13.06** | 46.86 / 29.69 / 26.21 | **Monitoring only.** Best raw P0−P1 gap of anything I tested (10.4pp) and it holds in 2019+ (17.2pp) and on the non-CI subset (17.28 / 11.64 / 8.95). But variant Z (`has_excl` + `has_logline` alone) measured **−0.17 / −0.18** macro-F1. TF-IDF's char_wb analyzer already sees `!` inside its 3-grams. Genuinely disappointing — this was my best hypothesis |
| `title_has_excl` | 0.32 / 0.08 / 0.02 | 0.52 / 0 / 0 | **Reject.** 27 issues corpus-wide |
| `title_has_caps_word` (ALL-CAPS ≥3 chars) | 15.05 / 15.05 / 12.78 | 12.20 / 13.74 / 13.84 | **Reject.** P0−P1 gap = **0.00pp**. The ALL-CAPS-as-urgency hypothesis is dead on this corpus — the caps words are mostly acronyms (`API`, `TLS`, `DNS`), not shouting |
| `has_logline` (klog/ISO-timestamp/log-level line) | 13.56 / 9.52 / 4.90 | 24.74 / 11.81 / 6.71 | **Monitoring only.** Monotone and survives the non-CI subset (13.21 / 8.92 / 5.68), but −0.17 in ablation and it is partly a CI-flake proxy |
| `has_expected` (expected/actual behavior section) | 11.43 / 11.43 / 7.30 | **21.60 / 30.84 / 53.67** | **Reject, and flag it.** The full-corpus number is flat (P0−P1 = 0.01pp) and the 2019+ number **inverts hard** — a completed template is a *backlog* marker, 53.7% vs 21.6%. That inversion is a k8s template-adoption artifact (the modern bug template forces the section), which makes it corpus-specific *and* unstable. Textbook trap |
| `has_envsec` (environment/version section) | 17.35 / 15.48 / 13.52 | 26.66 / 28.81 / 53.04 | **Reject.** Same inversion, same cause |
| `has_repro` (steps-to-reproduce) | 7.46 / 7.21 / 3.21 | 12.89 / 14.63 / 16.14 | **Reject.** P0−P1 = 0.25pp on the full corpus, inverts in 2019+ |
| `still_broken` / "regression from" / "used to work" | 2.08 / 1.72 / 0.74 | 2.26 / 2.73 / 0.84 | **Reject.** Fires on 1.5% of the corpus, P0−P1 gap 0.36pp. Right idea, no volume |
| `title_bracket` (title starts `[...]`) | **27.47 / 10.65 / 12.14** | 52.09 / 26.96 / 8.81 | **Reject — this one is a trap.** Second-strongest raw signal I found, and it is almost entirely the CI-flake cluster: on the **non-CI subset** it collapses to 6.55 / 2.90 / 1.93, and on the 2019+ non-CI slice it *inverts* (3.91 / 8.64 / 5.53). Repo-generic in form, k8s-specific in substance |
| `title_version` (title starts with a version number) | 0.95 / 0.47 / 0.23 | 1.22 / 0.26 / 0.00 | **Reject.** 71 issues corpus-wide |
| `title_component` (`component: rest of title`) | 9.31 / 6.45 / 8.72 | 3.14 / 5.11 / 9.85 | **Reject.** Non-monotone both eras |
| `code_ratio` (fraction of body inside fences) | mean 0.339 / 0.244 / 0.241 | — | **Monitoring only.** Separates P0 but not P1/P2; inside block F which measured −0.12 |
| `n_sentences` / readability proxy | median 1 / 3 / 3 | — | **Reject.** The P0 median of 1 sentence is a CI-flake artifact (machine-generated bodies have no sentence punctuation), not a readability signal |
| `title_has_q` (title is a question) | 0.36 / 0.82 / 1.32 | — | **Reject.** Right direction (questions skew backlog) but fires on 0.9% of the corpus |
| **Digit normalization of the text channel** | — | — | **ACCEPT — the one real addition.** See leakage re-audit |

**Block-level verdict on all new candidates together:** variant F (all 12 of them) measured
**−3.08 macro-F1** on the 80/20 split and −0.12 across 3 folds. It buys P0 recall (63.0 → 69.5) and
pays for it with P1 (56.0 → 48.9) and P2 (34.6 → 30.0). Under the proposed asymmetric cost matrix
that trade might actually be defensible — but it should be bought with the decision threshold on a
calibrated model, not by bolting on features that shift the prior sideways. **Reject the block.**

---

## Redundancy analysis — what TF-IDF already subsumes

| Derived feature | Subsumed by | Evidence |
|---|---|---|
| All 15 severity-lexicon flags | word(1,2) | **21/22 terms present in the fitted 55,579-term word vocabulary.** Only `segfault` missing (18 issues), covered by char n-grams |
| `has_code_block` | char_wb(3,5) | The literal token ``` ``` `` is a char 3-gram. The vectorizer has 114,793 char features; backtick-runs, indentation and `$`-prompt patterns are all in-band. Not separately proven, but the mechanism is direct |
| `has_stack_trace` | word(1,2) + char_wb | `goroutine` and `traceback` are both in the word vocabulary; `.go:NNN +0x` is a char-gram pattern |
| `has_url` | char_wb(3,5) | `http`, `://`, `.com` are all char 3-4 grams |
| `has_excl` | char_wb(3,5) | `!` participates in char n-grams (`char_wb` does not strip punctuation) |
| `has_logline` | char_wb(3,5) | klog prefixes (`E0921 14:`) are char-gram-shaped |
| `has_numbered_list` / `has_bullet_list` | char_wb(3,5) | `\n1. ` / `\n- ` are char 3-grams |
| length / word count | **NOT subsumed** | TF-IDF rows are L2-normalized, which deliberately removes document length. This is the one derived family that is genuinely orthogonal — and it still measured +0.06 / −0.32. Orthogonal and useless is a real outcome |
| `author_association` | **NOT subsumed** | Not in the text at all |

**Honest summary:** of the ten drafted derived features, **eight are mechanically duplicated by the
char or word analyzer**, and the two that are not (length, author_association) are the two that
measured closest to zero. The drafted feature list was, in effect, hand-encoding a prior over
features the vectorizer was already going to learn from data.

---

## Leakage re-audit — adversarial mode

### 1. Does body length leak triage effort? — Partially yes, and it is a reason to cut it

Median body chars: **P0 948 · P1 762 · P2 597.** Means 2,333 / 1,859 / 1,651. The gradient is real
and monotone. Two competing explanations:

- *Benign:* severe issues genuinely need more evidence (logs, repro, versions) to report.
- *Leaky:* reporters who already know an issue is severe invest more writing effort, so length is a
  proxy for the reporter's own prior — which is downstream of the same judgement that produced the
  label.

**I cannot separate these with this data — this is genuinely not measurable without the timeline
API** (was the body edited after triage? GitHub bodies are mutable and `updated_at ≠ created_at` for
100% of rows, so *some* of these bodies were edited post-triage and we cannot tell which). That
unfalsifiable-leak risk plus a measured contribution of +0.06 macro-F1 is an easy call: **cut it.**
Not worth defending a feature that might be a leak and definitely does not help.

Note the same suspicion partially applies to the raw text channel itself — a post-triage body edit
contaminates TF-IDF too. But we cannot ship a scorer without text, so that is an accepted,
documented residual risk, not an avoidable one. **Open question: fetch `body` edit history for a
sample of 200 issues via the timeline API and measure what fraction were edited after the priority
label was applied.** Not measured.

### 2. Is `has_code_block` a proxy for the banned CI-flake cluster? — Partly, but not entirely

Split by `kind/flake`/`kind/failing-test` membership:

| Subset | has_code_block P0/P1/P2 | has_stack_trace | median body_chars |
|---|---|---|---|
| CI-flake (n=3,373) | 62.77 / 65.27 / **71.58** | 2.82 / 3.75 / 1.96 | 982 / 1,018 / 682 |
| non-CI (n=8,584) | 45.68 / 34.49 / 31.20 | 5.56 / 2.82 / 1.08 | 896 / 674 / 562 |

Inside the CI-flake cluster the code-block signal **inverts** (P2 highest at 71.58%) — so the
full-corpus number 54.99/41.39/40.12 is a mixture of two opposite populations. That is a
Simpson's-paradox setup, and a model given `has_code_block` as a single scalar will fit the
mixture, not either population. On the non-CI subset the feature does retain honest separation
(45.68 / 34.49 / 31.20), so it is **not purely** a CI artifact — but the mixture instability is one
more reason it does not belong as a raw scalar.

The body-length gradient survives inside both subsets (896/674/562 non-CI), so length is not merely
a CI-flake proxy either. It is the triage-effort question above that kills it, not this one.

### 3. NEW LEAK FOUND — calendar tokens in the text channel

This is the finding that justifies the whole review. Top-30 P0 word features from
`LogisticRegression` on raw TF-IDF, 80/20 time split:

> `failing, k8s, e2e, pr, `**`sep`**`, node kubernetes, revert, cni, gke, `**`v10`**`, `**`06`**`,
> is broken, from the, broken, p0, csi, are failing, io, fix, were, logs kubernetes, `**`11e7`**`,
> content, `**`2017`**`, `**`08`**`, blocking, due to, due, deployment should, kubeadm`

Top-20 P2 features include `2015`, `serial release`, `ci kubernetes`, `conformance kubernetes`.

Two separate problems, both invisible in the prior pass's leakage table:

- **Calendar/era tokens** (`2017`, `2015`, `sep`, `06`, `08`) are being used as class evidence. The
  prior memo correctly bans `created_at` as a model feature — but the *date is also written in the
  text*, in log timestamps, version strings, and pasted CI output. Given the measured 5%→42%→10%
  swing in the P0 prior across years, a year token is a direct read on the label prior. Banning the
  `created_at` field while leaving `2017` in the vocabulary closes the front door and leaves the
  window open.
- **Build/run identifiers** (`11e7`, `v10`) are per-CI-run nonces. They cannot generalize to
  anything, in any repo.

**Measured fix cost:** replacing all digits with `#`, hex runs ≥7 with `HASH`, and dropping month
names costs **−0.03 macro-F1** across 3 folds (45.17 → 45.14) and **improves the non-CI slice
+0.60** (44.04 → 44.64). Free, and it removes an entire class of non-generalizing feature.
**Make it mandatory preprocessing.**

I also tested stripping an explicit k8s-vocabulary list (`kubernetes|k8s|kubelet|etcd|apiserver|
gke|kubeadm|kubemark|e2e|prow|sig-*|csi|cni|…`): **−0.34 macro-F1** (45.17 → 44.83), i.e. within
noise. That is a *useful* number for the repo-generic constraint — it means the k8s-specific
vocabulary is carrying almost none of the in-domain performance, so removing it costs nothing and
could plausibly help transfer. **Recommendation: do not ship the k8s stoplist in v0** (it is
repo-specific machinery in a repo-generic model, and it is a maintenance list nobody will update),
**but run it as an arm in the v0.5 cross-repo experiment** — that is where its value, if any, will
show up.

### 4. Re-confirmed, no new issues

The serving-contract whitelist itself holds. Nothing I recommend reads `labels`, `milestone`,
`assignee`, `comments`, `reactions`, `state`, `closed_at`, or `updated_at`. `ci_label` is used in
this memo **only** as an offline stratification key computed from `labels` — it is a
*measurement* variable, and it must never enter `build_features()`. The CI-flake stratified eval the
prior memo mandates has to derive its strata from the eval set's stored labels, not from a feature.
**Write that down in the evaluation module, or someone will helpfully "add" it as a feature.**

---

## Self-verification pass

I re-ran or re-derived every number in this memo. What that caught:

1. **The 11,841 error.** Found by recomputing the multi-priority decomposition independently
   (`92 = 34 core-conflicts + 56 longterm + 2 awaiting`) rather than trusting the arithmetic in the
   prior memo. Corrected to 11,899.
2. **The dense-scaling artifact.** My first ablation standard-scaled the derived block and reported
   every variant as a loss, including −1.37 for the lexicon. Re-running with min-max scaling moved
   several deltas by ~0.4 F1 and flipped the sign on three of them. **I would have cut the right
   features for the wrong reason.** Both scalings are reported above; the conclusion (noise-level)
   survives either way, but the confidence in individual per-block signs does not.
3. **Single-split over-confidence.** The 80/20 split alone said "everything hurts." Three
   forward-chaining folds said "nothing is distinguishable from noise, fold variance is ±10 F1."
   The second is the honest reading and it is what the memo now says. I explicitly did **not**
   report the single-split table alone.
4. **The `has_excl` disappointment.** I proposed it, measured a genuinely good class-conditional
   separation (10.4pp P0−P1, holds on non-CI and in 2019+), then ablated it and got −0.17. I am
   reporting the negative result rather than shipping my own proposal on the strength of the
   contingency table — which is exactly the error I am criticizing in the drafted lexicon.
5. **Contract compliance re-check.** Every recommended feature reads only `title`, `body`,
   `author_association`. `created_at` appears only as a sort key. `ci_label` is quarantined to
   evaluation. ✅
6. **Redundancy claims re-check.** The lexicon-subsumption claim is backed by an actual vocabulary
   membership test (21/22), not asserted. The `has_code_block` / `has_url` / `has_excl`
   subsumption claims are **mechanistic arguments, not ablations** — I have said so in the table
   rather than implying they were measured.

### Still uncertain — flag these, do not treat as settled

- **char_wb(3,5) has not been ablated against word-only.** It is the single largest untested
  assumption in the v0 design (114,793 of the 170,372 features). Run it in the v0 experiment.
- **Fold variance of ±10 macro-F1 makes every conclusion here low-power.** Fold 0 (test window
  ~2016) scores 33; fold 2 scores 49. With that variance, a true +1.0 F1 effect from derived
  features is not detectable at n=11,957. My claim is "not shown to help," not "proven useless."
- **Whether the body text was edited after triage** — unfalsifiable without the timeline API.
  Affects the text channel itself, not just the length feature.
- **Cross-repo transfer of anything here.** Zero holdout data has been downloaded. Every
  repo-genericity judgement in this memo is a judgement about *mechanism*, not a measurement.
- **`C` regularization was left at 1.0 everywhere and never tuned.** A tuned C might change the
  derived-feature deltas, since the dense block competes with 170k sparse features for the same
  penalty budget. Not measured.
- **The 41.9% vs 50.9% CI-title discrepancy** with the prior memo is unresolved. Someone should
  reconcile the two regexes before either number is quoted again.

---

*All figures produced by scratch scripts run against `data/raw/kubernetes-kubernetes/*.jsonl` on
2026-08-25 with scikit-learn 1.9.0. Scripts live in the session scratchpad, not committed — they are
measurement, not pipeline code.*
