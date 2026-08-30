"""Shared synthetic-data helpers for tests/evaluation/*.

Not a conftest.py fixture module -- plain importable helpers, mirroring the
inline `make_feature_row`/`make_rows` helpers already used by
tests/training/test_train_baseline.py, just shared across this subpackage's
three test files since all three need the same tiny BaselineArtifact/
FeatureRow scaffolding.
"""

from datetime import UTC, datetime, timedelta

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
