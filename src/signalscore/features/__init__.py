"""Feature pipeline stage.

Builds three feature families from collected issues: text signals (title/body
length, stack-trace presence, error-text presence, urgency language), GitHub
metadata (author association, label history, comment/reaction count,
time-since-open, issue-template-used), and BGE-large embeddings reused from
the sibling DocMind project rather than rebuilt here. See
docs/signalscore-design.md, the feature set section, for the full feature
table.
"""
