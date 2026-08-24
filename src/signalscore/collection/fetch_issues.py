"""Fetch kubernetes/kubernetes issues by priority/* label into per-label JSONL files.

Reproducible dataset pull for training. Run via:
    uv run python -m signalscore.collection.fetch_issues

Uses the REST "list issues" endpoint, not Search — Search caps at 1000
results/query, which `priority/backlog` alone (~4.9K issues) exceeds. See
docs/signalscore-design.md, Data Source Strategy, for why this repo/label
set is the ground truth source.
"""

import argparse
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "kubernetes/kubernetes"
DEFAULT_LABELS = (
    "priority/critical-urgent",
    "priority/important-soon",
    "priority/backlog",
)
REQUEST_TIMEOUT = 30
PER_PAGE = 100


def _get(session: requests.Session, url: str, params: dict[str, Any] | None) -> requests.Response:
    """GET with a single retry after sleeping out a primary rate-limit block."""
    resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset_at = int(resp.headers["X-RateLimit-Reset"])
        time.sleep(max(reset_at - time.time(), 0) + 1)
        resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


def iter_label_issues(session: requests.Session, repo: str, label: str) -> Iterator[dict[str, Any]]:
    """Yield raw issue dicts for one label, paginating via the Link header, skipping PRs.

    One label per request (not comma-joined): GitHub's `labels` param is
    AND-semantics across a comma list, and we need OR across the three
    priority labels — matched by writing one JSONL file per label anyway.
    """
    url: str | None = f"{GITHUB_API}/repos/{repo}/issues"
    params: dict[str, Any] | None = {
        "labels": label,
        "state": "all",
        "per_page": PER_PAGE,
        "direction": "asc",
    }
    while url:
        resp = _get(session, url, params)
        for item in resp.json():
            if "pull_request" not in item:
                yield item
        url = resp.links.get("next", {}).get("url")
        params = None  # the next URL already carries the query string


def fetch_label(
    session: requests.Session, repo: str, label: str, out_path: Path, *, force: bool
) -> None:
    """Write one label's issues to `out_path` as JSONL.

    Streams to a `.jsonl.tmp` file and renames atomically on completion, so a
    crash mid-pull never leaves a partial file mistaken for a finished one.
    Skips entirely (no HTTP calls) if `out_path` already exists and `force`
    is not set — makes re-runs resumable at the per-label granularity.
    """
    if out_path.exists() and not force:
        print(f"skip {label}: {out_path} already exists")
        return
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    count = 0
    with tmp_path.open("w") as f:
        for issue in iter_label_issues(session, repo, label):
            issue["_queried_label"] = label
            f.write(json.dumps(issue) + "\n")
            count += 1
    tmp_path.rename(out_path)
    print(f"{label}: {count} issues -> {out_path}")


def build_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN not set — add it to .env (see .env.example)")

    session = build_session(token)
    repo_dir = args.output_dir / args.repo.replace("/", "-")
    repo_dir.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        out_path = repo_dir / f"{label.replace('/', '-')}.jsonl"
        fetch_label(session, args.repo, label, out_path, force=args.force)


if __name__ == "__main__":
    main()
