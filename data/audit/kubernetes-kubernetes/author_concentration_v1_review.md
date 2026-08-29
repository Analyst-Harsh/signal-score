# Manual review: author_concentration_v1.json

Confirmed: zero of `k8s-github-robot`'s issues land in test. It's a legacy
bot account (predates the more recent `k8s-ci-robot`) that was active
earlier in the corpus's history and simply stopped being used before the
time-based test cutoff. Its label mix isn't a trivial single-label pattern
either — 60% backlog, 24% critical-urgent, 16% important-soon — so it's not
literally "same text -> same label" degenerate noise.

## Verdict on all three flags

- **Train (HHI 0.038, top1 17.1%) and val (HHI 0.043, top1 18.3%):** fully
  explained by this one bot. Not a leakage risk — leakage would require the
  concentrated author's text to also appear in test, and it doesn't. What it
  does represent is a real train/test distributional shift (a legacy filer
  active in the older 80% of the corpus, silent in the most recent 20%),
  which is a consequence of the time-based split design already locked in,
  not a new problem to fix.
- **Test (HHI 0.011, top1 5.1%):** genuinely diffuse, human-authored, no
  concerning concentration.

## Decision: no dataset changes needed

Dropping or downweighting the bot's issues would just be throwing away real
training signal to chase a metric that isn't actually measuring risk here.
The one real action item is calibration, not remediation: the
`hhi_threshold=0.02` / `top1_threshold=0.05` defaults in
`audit_author_concentration` are too tight for any corpus with an automated
filer — worth raising them (or excluding known bot logins from the
calculation) so future runs don't cry wolf on this same, already-understood
pattern. Left as placeholders per the code's own docstring; not changed in
this review.
