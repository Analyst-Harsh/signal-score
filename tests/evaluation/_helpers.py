"""Shared synthetic-data helpers for tests/evaluation/*.

Not a conftest.py fixture module -- plain importable helpers, mirroring the
inline `make_feature_row`/`make_rows` helpers already used by
tests/training/test_train_baseline.py, just shared across this subpackage's
three test files since all three need the same tiny BaselineArtifact/
FeatureRow scaffolding.
"""

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from signalscore.evaluation.contract import FROZEN_EVAL_TAG
from signalscore.features.labels import Priority
from signalscore.features.schema import FeatureRow
from signalscore.training.train_baseline import (
    BaselineArtifact,
    build_feature_matrix,
    compute_feature_std,
    extract_labels,
    fit_vectorizers,
    train_classifier,
)

CLASS_TEXTS = {
    Priority.P0_CRITICAL: "critical outage pod crash cluster panic urgent failure",
    Priority.P1_SOON: "important bug slow response needs investigation soon fix",
    Priority.P2_BACKLOG: "minor cosmetic typo backlog low priority cleanup task",
}


def make_feature_row(
    issue_number: int,
    text: str,
    label: Priority = Priority.P2_BACKLOG,
    is_member_plus: bool = False,
) -> FeatureRow:
    return FeatureRow(
        issue_number=issue_number,
        created_at=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=issue_number),
        label=label,
        text=text,
        is_member_plus=is_member_plus,
        title_chars=1,
        body_chars=1,
        body_words=1,
        has_code_block=False,
        has_stack_trace=False,
        has_url=False,
        has_logline=False,
        has_excl=False,
        code_ratio=0.0,
        any_lex=False,
        is_ci_flake_shaped=False,
    )


def make_rows(n_per_class: int, start_issue: int = 0) -> list[FeatureRow]:
    """Enough repeated shared vocabulary per class to clear the locked spec's
    min_df=3 (word) / min_df=5 (char_wb) thresholds.
    """
    rows: list[FeatureRow] = []
    issue = start_issue
    for label, text in CLASS_TEXTS.items():
        for i in range(n_per_class):
            rows.append(
                make_feature_row(
                    issue, f"{text} variant {i}", label=label, is_member_plus=(i % 2 == 0)
                )
            )
            issue += 1
    return rows


def init_frozen_git_dvc_repo(tmp_path: Path, content: str = "hello v1") -> Path:
    """Builds a real tmp_path git+dvc repo: init both, write+track one file,
    commit, tag it FROZEN_EVAL_TAG. Returns the tracked file's path.

    The real eval set is DVC-tracked and gitignored (only its `.dvc` pointer
    is git-committed) -- it lives in a machine-local DVC remote and is never
    present on a CI runner. Any test exercising `ContractStep`/`run_gate.py`
    for real must point at a throwaway repo like this one instead, per the
    plan's own guidance for testing this real-subprocess integration.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    eval_file = data_dir / "eval_set_v1.jsonl"
    eval_file.write_text(content)

    def run(*args: str) -> None:
        subprocess.run(args, cwd=tmp_path, check=True, capture_output=True, text=True)  # noqa: S603

    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    run("dvc", "init", "-q")
    run("dvc", "add", "-q", str(eval_file.relative_to(tmp_path)))
    run("git", "add", "data/eval_set_v1.jsonl.dvc", "data/.gitignore")
    run("git", "commit", "-q", "-m", "init")
    run("git", "tag", FROZEN_EVAL_TAG)
    return eval_file


def build_artifact(rows: list[FeatureRow]) -> BaselineArtifact:
    """Fits fresh vectorizers + a classifier on `rows`, via train_baseline.py's
    own functions -- never a reimplementation of training logic.
    """
    vectorizers = fit_vectorizers([row.text for row in rows])
    x = build_feature_matrix(rows, vectorizers)
    y = extract_labels(rows)
    model = train_classifier(x, y)
    classes = [str(c) for c in model.classes_]  # pyright: ignore
    return BaselineArtifact(
        word_vectorizer=vectorizers.word,
        char_vectorizer=vectorizers.char,
        model=model,
        classes=classes,
        feature_std=compute_feature_std(x),
    )
