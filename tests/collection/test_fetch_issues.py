"""Tests for signalscore.collection.fetch_issues."""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import requests

from signalscore.collection.fetch_issues import fetch_label, iter_label_issues


def _response(items: list[dict[str, Any]], next_url: str | None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = items
    resp.links = {"next": {"url": next_url}} if next_url else {}
    resp.raise_for_status.return_value = None
    return resp


def test_iter_label_issues_filters_pull_requests() -> None:
    mock_session = MagicMock(spec=requests.Session)
    issue = {"number": 1, "title": "a real issue"}
    pr = {"number": 2, "title": "a PR", "pull_request": {"url": "..."}}
    mock_session.get.return_value = _response([issue, pr], next_url=None)

    result = list(
        iter_label_issues(
            cast(requests.Session, mock_session), "kubernetes/kubernetes", "priority/backlog"
        )
    )

    assert result == [issue]


def test_iter_label_issues_follows_pagination() -> None:
    mock_session = MagicMock(spec=requests.Session)
    page1_issue = {"number": 1}
    page2_issue = {"number": 2}
    mock_session.get.side_effect = [
        _response([page1_issue], next_url="https://api.github.com/next-page"),
        _response([page2_issue], next_url=None),
    ]

    result = list(
        iter_label_issues(
            cast(requests.Session, mock_session), "kubernetes/kubernetes", "priority/backlog"
        )
    )

    assert result == [page1_issue, page2_issue]
    assert mock_session.get.call_count == 2


def test_fetch_label_skips_existing_file_without_force(tmp_path: Path) -> None:
    mock_session = MagicMock(spec=requests.Session)
    out_path = tmp_path / "priority-backlog.jsonl"
    out_path.write_text('{"number": 1}\n')

    fetch_label(
        cast(requests.Session, mock_session),
        "kubernetes/kubernetes",
        "priority/backlog",
        out_path,
        force=False,
    )

    mock_session.get.assert_not_called()


def test_fetch_label_writes_jsonl_and_cleans_up_tmp_file(tmp_path: Path) -> None:
    mock_session = MagicMock(spec=requests.Session)
    issue = {"number": 1, "title": "a real issue"}
    mock_session.get.return_value = _response([issue], next_url=None)
    out_path = tmp_path / "priority-backlog.jsonl"

    fetch_label(
        cast(requests.Session, mock_session),
        "kubernetes/kubernetes",
        "priority/backlog",
        out_path,
        force=False,
    )

    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {**issue, "_queried_label": "priority/backlog"}
    assert not out_path.with_suffix(".jsonl.tmp").exists()
