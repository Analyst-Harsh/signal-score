# Signal Score

Classical ML scoring service that predicts GitHub issue priority from GitHub
metadata, text signals, and BGE embeddings — trained on `kubernetes/kubernetes`
issue labels, with an explicit promotion-gate/MLOps loop (train → MLflow
registry → gated eval vs. production → FastAPI serving → Evidently drift
monitoring → GitHub-Actions-driven retraining).

Full design lives in [`docs/signalscore-design.md`](docs/signalscore-design.md)
and [`docs/signalscore-architecture.mermaid`](docs/signalscore-architecture.mermaid).
Agent-facing conventions and guardrails live in [`CLAUDE.md`](CLAUDE.md).

## Quickstart

```bash
uv sync --group dev
make check
```

## License

MIT — see [`LICENSE`](LICENSE).
