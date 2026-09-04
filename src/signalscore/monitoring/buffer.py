"""Live-traffic buffer -- Evidently's "current" dataset.

Every `/score` request appends one line here: the input columns
`monitoring.reference.INPUT_DRIFT_COLUMNS` defines, plus the model's own
predicted class and a timestamp. This is the gap identified while planning
this module -- `/score` itself persists nothing else about a request (see
serving/app.py's own docstring: only a structlog line). Single-writer
assumption: fine for the current single-process serving MVP.
ponytail: move to per-worker files or a real queue if serving ever runs
multi-worker.

Write-failure handling lives at the call site (serving/app.py's `/score`
handler), not here -- a disk-write hiccup on the monitoring path must never
break scoring itself, but that's a concern of the caller, not of this
function's own contract.

ponytail: no rotation or size cap on the buffer file itself -- it grows for
the life of the service. `load_live_traffic`'s `tail` window keeps a drift
check's "current" dataset bounded to recent traffic (so old, non-drifted
history can't dilute a real recent shift toward a permanent false "no
drift"), but the file on disk keeps growing; revisit with a real rotation
policy if disk usage or read cost ever becomes real.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from signalscore.features.schema import FeatureRow
from signalscore.monitoring.reference import INPUT_DRIFT_COLUMNS

_REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_TRAFFIC_PATH = _REPO_ROOT / "data/monitoring/live_traffic.jsonl"

_BUFFER_COLUMNS: tuple[str, ...] = (*INPUT_DRIFT_COLUMNS, "predicted_class", "recorded_at")

# Fixed row count, not a time window -- the simplest bound that stops the
# "current" dataset from being diluted by the entire lifetime of traffic.
# Tune (or switch to a real time window) once real traffic volume/velocity
# is known.
DEFAULT_TAIL = 500


def record_live_traffic(
    row: FeatureRow, predicted_class: str, path: Path = LIVE_TRAFFIC_PATH
) -> None:
    entry = {
        **{col: getattr(row, col) for col in INPUT_DRIFT_COLUMNS},
        "predicted_class": predicted_class,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_live_traffic(
    path: Path = LIVE_TRAFFIC_PATH, tail: int | None = DEFAULT_TAIL
) -> pd.DataFrame:
    """Empty (but correctly-columned) frame when nothing's been recorded
    yet -- callers (DriftChecker) can run against a fresh deployment without
    a special-cased "no traffic yet" branch of their own.

    `tail` keeps only the most recently recorded `tail` rows (None = every
    row ever recorded) -- see this module's own ponytail note on why an
    unbounded "current" dataset would dilute a real recent drift. A
    malformed trailing line (e.g. a process killed mid-write) is skipped,
    not raised -- a corrupt last line must not crash every drift check from
    then on.
    """
    if not path.exists():
        return pd.DataFrame(columns=_BUFFER_COLUMNS)
    entries: list[dict[str, object]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if tail is not None:
        entries = entries[-tail:]
    return pd.DataFrame(entries, columns=_BUFFER_COLUMNS)
