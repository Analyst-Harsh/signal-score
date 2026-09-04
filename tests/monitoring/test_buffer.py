from pathlib import Path

from signalscore.monitoring.buffer import load_live_traffic, record_live_traffic
from tests.evaluation._helpers import make_feature_row


def test_load_live_traffic_returns_an_empty_correctly_columned_frame_when_nothing_recorded(
    tmp_path: Path,
) -> None:
    frame = load_live_traffic(tmp_path / "never_written.jsonl")

    assert len(frame) == 0
    assert "predicted_class" in frame.columns
    assert "text" in frame.columns


def test_record_then_load_round_trips_input_columns_and_prediction(tmp_path: Path) -> None:
    path = tmp_path / "live_traffic.jsonl"
    row = make_feature_row(1, "pod crash cluster panic", is_member_plus=True)

    record_live_traffic(row, predicted_class="P0_critical", path=path)
    record_live_traffic(row, predicted_class="P1_soon", path=path)

    frame = load_live_traffic(path)

    assert len(frame) == 2
    assert list(frame["predicted_class"]) == ["P0_critical", "P1_soon"]
    assert list(frame["text"]) == [row.text, row.text]
    assert list(frame["is_member_plus"]) == [True, True]
    assert "recorded_at" in frame.columns


def test_record_live_traffic_appends_without_truncating(tmp_path: Path) -> None:
    path = tmp_path / "live_traffic.jsonl"
    row = make_feature_row(1, "minor cosmetic typo")

    for _ in range(3):
        record_live_traffic(row, predicted_class="P2_backlog", path=path)

    assert len(load_live_traffic(path)) == 3


def test_load_live_traffic_tail_keeps_only_the_most_recent_rows(tmp_path: Path) -> None:
    path = tmp_path / "live_traffic.jsonl"
    row = make_feature_row(1, "pod crash cluster panic")

    for predicted_class in ["P0_critical", "P1_soon", "P2_backlog"]:
        record_live_traffic(row, predicted_class=predicted_class, path=path)

    frame = load_live_traffic(path, tail=2)

    assert list(frame["predicted_class"]) == ["P1_soon", "P2_backlog"]


def test_load_live_traffic_skips_a_corrupt_trailing_line(tmp_path: Path) -> None:
    path = tmp_path / "live_traffic.jsonl"
    row = make_feature_row(1, "pod crash cluster panic")
    record_live_traffic(row, predicted_class="P0_critical", path=path)

    with path.open("a") as f:
        f.write('{"predicted_class": "P1_soon", "recorded_at":')  # truncated mid-write

    frame = load_live_traffic(path)

    assert len(frame) == 1
    assert frame["predicted_class"].iloc[0] == "P0_critical"
