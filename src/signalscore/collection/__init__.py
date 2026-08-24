"""Data collection stage.

Pulls issues from `kubernetes/kubernetes` via the GitHub REST API, using the
maintainer-assigned `priority/critical-urgent`, `priority/important-soon`, and
`priority/backlog` labels as ground truth. Optionally supplements training
volume with issues from `kubernetes-sigs/*` repos that share the identical
Prow-bot-enforced `priority/*` taxonomy — never blend in a repo using a
different label taxonomy. Also fetches issues from a different-taxonomy repo
(Rust or VS Code) as a zero-shot generalization holdout set; this set is used
only to measure cross-taxonomy generalization and is never trained on. See
docs/signalscore-design.md, the data source strategy section, for the full
rationale.
"""
