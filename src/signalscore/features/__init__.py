"""Feature pipeline stage.

Builds the locked v0 feature contract via `build_features(title, body,
author_association)` in `pipeline.py`: one TF-IDF-ready text channel
(`text.py`, with mandatory digit normalization) and one metadata bit
(`metadata.py`, `is_member_plus`). A second, diagnostic-only block (code/URL/
stack-trace/log-line/lexicon flags, lengths) is computed alongside for
drift-monitoring and stratification but never enters the model matrix — see
docs/priority-signal-decision.md and docs/v0-feature-selection.md, which
supersede this module's older BGE/broad-metadata description. The label
taxonomy and its repo-parametrized YAML mapping live in `labels.py`;
`loading.py` handles raw JSONL ingestion, dedup, and row-dropping; `schema.py`
defines the assembled matrix's row contract.
"""
