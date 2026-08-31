"""Tests for signalscore.training.train_baseline."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import joblib
import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from scipy import sparse
from sklearn.metrics import (  # pyright: ignore[reportMissingTypeStubs]
    average_precision_score,  # pyright: ignore[reportUnknownVariableType]
    brier_score_loss,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.preprocessing import (
    label_binarize,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)

from signalscore.features.labels import Priority
from signalscore.features.schema import FeatureRow
from signalscore.registry import DEFAULT_MODEL_NAME
from signalscore.training.strategies import compute_feature_std, train_classifier
from signalscore.training.train_baseline import (
    BaselineArtifact,
    FittedVectorizers,
    build_feature_matrix,
    compute_metrics,
    fit_vectorizers,
    format_config_label,
    format_experiments_row,
    load_feature_rows,
    main,
    parse_args,
    run_training_pipeline,
    train_and_evaluate,
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


def write_rows(rows: list[FeatureRow], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def test_load_feature_rows_round_trips(tmp_path: Path) -> None:
    rows = [make_feature_row(1, "alpha"), make_feature_row(2, "beta")]
    path = tmp_path / "rows.jsonl"
    write_rows(rows, path)

    loaded = load_feature_rows(path)

    assert loaded == rows


def test_fit_vectorizers_builds_vocabulary_from_given_texts_only() -> None:
    train_texts = [f"kubernetes pod crash cluster variant {i}" for i in range(6)]
    held_out_text = "kubernetes pod crash cluster zzzunique"

    vectorizers = fit_vectorizers(train_texts)

    word_vocabulary: dict[str, int] = vectorizers.word.vocabulary_  # pyright: ignore
    assert "zzzunique" not in word_vocabulary
    assert "pod" in word_vocabulary
    assert held_out_text not in train_texts


def test_fit_vectorizers_default_kwargs_reproduce_locked_spec_config() -> None:
    """Regression guard for Experiment 2 (docs/candidate_models/overview.md):
    calling fit_vectorizers with no overrides must still produce the exact
    locked-spec config -- word (1,2)/min_df=3/sublinear_tf=True, char_wb
    (3,5)/min_df=5/max_features=300_000/sublinear_tf=True.
    """
    vectorizers = fit_vectorizers([f"pod crash cluster variant {i}" for i in range(6)])

    assert vectorizers.word.ngram_range == (1, 2)  # pyright: ignore
    assert vectorizers.word.min_df == 3  # pyright: ignore
    assert vectorizers.word.sublinear_tf is True  # pyright: ignore
    assert vectorizers.char.ngram_range == (3, 5)  # pyright: ignore
    assert vectorizers.char.min_df == 5  # pyright: ignore
    assert vectorizers.char.max_features == 300_000  # pyright: ignore
    assert vectorizers.char.sublinear_tf is True  # pyright: ignore


def test_fit_vectorizers_word_kwargs_are_overridable() -> None:
    vectorizers = fit_vectorizers(
        [f"pod crash cluster variant {i}" for i in range(6)],
        word_ngram_range=(1, 1),
        word_min_df=1,
        word_sublinear_tf=False,
    )

    assert vectorizers.word.ngram_range == (1, 1)  # pyright: ignore
    assert vectorizers.word.min_df == 1  # pyright: ignore
    assert vectorizers.word.sublinear_tf is False  # pyright: ignore
    # char vectorizer is never tunable, even when word kwargs change.
    assert vectorizers.char.ngram_range == (3, 5)  # pyright: ignore


def test_build_feature_matrix_transforms_val_without_refitting() -> None:
    train_rows = make_rows(n_per_class=6)
    vectorizers = fit_vectorizers([row.text for row in train_rows])
    word_vocabulary = cast("dict[str, int]", vectorizers.word.vocabulary_)  # pyright: ignore
    char_vocabulary = cast("dict[str, int]", vectorizers.char.vocabulary_)  # pyright: ignore
    expected_columns = len(word_vocabulary) + len(char_vocabulary) + 1

    val_rows = [make_feature_row(999, "brandnewwordneverseenbefore in this corpus at all")]
    matrix = build_feature_matrix(val_rows, vectorizers)

    assert matrix.shape == (1, expected_columns)


def test_build_feature_matrix_appends_is_member_plus_as_last_column() -> None:
    train_rows = make_rows(n_per_class=6)
    vectorizers = fit_vectorizers([row.text for row in train_rows])

    rows = [
        make_feature_row(1, CLASS_TEXTS[Priority.P0_CRITICAL], is_member_plus=True),
        make_feature_row(2, CLASS_TEXTS[Priority.P1_SOON], is_member_plus=False),
    ]
    matrix = build_feature_matrix(rows, vectorizers).toarray()

    assert matrix[0, -1] == 1.0
    assert matrix[1, -1] == 0.0


def test_compute_feature_std_matches_numpy_column_std_on_a_dense_equivalent() -> None:
    dense = np.array([[1.0, 0.0, 5.0], [0.0, 0.0, 5.0], [1.0, 1.0, 5.0], [0.0, 1.0, 5.0]])
    x = sparse.csr_matrix(dense)  # pyright: ignore

    std = compute_feature_std(x)

    # ddof=0 (population std) matches E[x^2] - E[x]^2, and the constant
    # last column (always 5.0, like a feature with zero real spread) must
    # come out as std 0, not blow up or go slightly negative under sqrt.
    assert std == pytest.approx(dense.std(axis=0, ddof=0))
    assert std[2] == pytest.approx(0.0)


def test_train_classifier_default_kwargs_reproduce_locked_spec_config() -> None:
    """Regression guard: no overrides must still produce today's exact
    LogisticRegression config (C=1.0, l1_ratio=0.0 i.e. "l2", solver="lbfgs").
    `penalty` itself is deprecated in sklearn 1.9+ (removed in 1.10) and is
    never passed to the constructor anymore -- l1_ratio is the real signal.
    """
    train_rows = make_rows(n_per_class=6)
    vectorizers = fit_vectorizers([row.text for row in train_rows])
    x_train = build_feature_matrix(train_rows, vectorizers)
    y_train = np.array([row.label.value for row in train_rows])

    model = train_classifier(x_train, y_train)

    assert pytest.approx(1.0) == model.C  # pyright: ignore
    assert pytest.approx(0.0) == model.l1_ratio  # pyright: ignore
    assert model.solver == "lbfgs"  # pyright: ignore
    assert model.class_weight == "balanced"  # pyright: ignore


def test_train_classifier_l1_penalty_selects_a_compatible_solver() -> None:
    """lbfgs (the default solver) only supports l1_ratio=0 ("l2") -- "l1"
    must switch to a solver that actually supports this 3-class problem
    rather than raising at fit time (liblinear does not: "does not support
    multiclass classification (n_classes >= 3)" on this repo's installed
    sklearn).
    """
    train_rows = make_rows(n_per_class=6)
    vectorizers = fit_vectorizers([row.text for row in train_rows])
    x_train = build_feature_matrix(train_rows, vectorizers)
    y_train = np.array([row.label.value for row in train_rows])

    model = train_classifier(x_train, y_train, penalty="l1")

    assert pytest.approx(1.0) == model.l1_ratio  # pyright: ignore
    assert model.solver == "saga"  # pyright: ignore
    model.predict(x_train)  # pyright: ignore -- fit actually succeeded on 3 classes


def test_train_classifier_is_deterministic_across_separate_calls() -> None:
    """Regression guard for Experiment 2's two-step flow: tune_baseline.py
    evaluates a config once, then a human re-runs train_baseline.py's CLI
    separately with the winning flags. Those two independent fits of the
    SAME config must produce identical predictions -- saga (the l1 solver)
    shuffles data stochastically, so without a fixed random_state the second
    fit could silently diverge from the config the sweep actually measured.
    """
    train_rows = make_rows(n_per_class=6)
    vectorizers = fit_vectorizers([row.text for row in train_rows])
    x_train = build_feature_matrix(train_rows, vectorizers)
    y_train = np.array([row.label.value for row in train_rows])

    first = train_classifier(x_train, y_train, penalty="l1")
    second = train_classifier(x_train, y_train, penalty="l1")

    assert (first.predict(x_train) == second.predict(x_train)).all()  # pyright: ignore
    assert np.allclose(first.coef_, second.coef_)  # pyright: ignore


def test_compute_metrics_on_a_hand_verified_tiny_case() -> None:
    classes = [Priority.P0_CRITICAL.value, Priority.P1_SOON.value, Priority.P2_BACKLOG.value]
    y_true = np.array(
        ["P0_critical", "P0_critical", "P1_soon", "P1_soon", "P2_backlog", "P2_backlog"]
    )
    y_pred = np.array(["P0_critical", "P1_soon", "P1_soon", "P1_soon", "P2_backlog", "P0_critical"])
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ]
    )

    metrics = compute_metrics(y_true, y_pred, y_proba, classes)

    # Hand-verified from the confusion matrix above:
    # P0 precision=0.5 recall=0.5 f1=0.5; P1 precision=2/3 recall=1.0 f1=0.8;
    # P2 precision=1.0 recall=0.5 f1=2/3.
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["per_class_f1"]["P0_critical"] == pytest.approx(0.5)
    assert metrics["per_class_f1"]["P1_soon"] == pytest.approx(0.8)
    assert metrics["per_class_f1"]["P2_backlog"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)

    expected_binarized: NDArray[np.float64] = label_binarize(  # pyright: ignore
        y_true, classes=classes
    )
    expected_pr_auc = float(
        average_precision_score(expected_binarized, y_proba, average="macro")  # pyright: ignore
    )
    assert metrics["pr_auc_macro"] == pytest.approx(expected_pr_auc)

    expected_brier = np.mean(
        [
            brier_score_loss((y_true == cls).astype(int), y_proba[:, i])
            for i, cls in enumerate(classes)
        ]
    )
    assert metrics["brier_macro"] == pytest.approx(expected_brier)


def test_compute_metrics_minority_f1_equals_p0_critical_per_class_f1() -> None:
    classes = [Priority.P0_CRITICAL.value, Priority.P1_SOON.value, Priority.P2_BACKLOG.value]
    y_true = np.array(["P0_critical", "P1_soon", "P2_backlog"])
    y_pred = np.array(["P0_critical", "P1_soon", "P0_critical"])
    y_proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.4, 0.3, 0.3]])

    metrics = compute_metrics(y_true, y_pred, y_proba, classes)

    assert metrics["minority_f1"] == metrics["per_class_f1"]["P0_critical"]


def test_compute_metrics_follows_passed_class_order_not_alphabetical() -> None:
    """Closes the class-ordering risk flagged in review: compute_metrics must use
    whatever `classes` list it is given for both binarizing y_true and indexing
    y_proba's columns -- never a silently-assumed alphabetical order.
    """
    y_true = np.array(
        ["P0_critical", "P0_critical", "P1_soon", "P1_soon", "P2_backlog", "P2_backlog"]
    )
    y_pred = np.array(["P0_critical", "P1_soon", "P1_soon", "P1_soon", "P2_backlog", "P0_critical"])
    classes_a = [Priority.P0_CRITICAL.value, Priority.P1_SOON.value, Priority.P2_BACKLOG.value]
    y_proba_a = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ]
    )
    # Same information, columns/labels reordered to [P1, P0, P2] -- as if the
    # fitted model's classes_ had come out in a different order.
    classes_b = [Priority.P1_SOON.value, Priority.P0_CRITICAL.value, Priority.P2_BACKLOG.value]
    y_proba_b = y_proba_a[:, [1, 0, 2]]

    metrics_a = compute_metrics(y_true, y_pred, y_proba_a, classes_a)
    metrics_b = compute_metrics(y_true, y_pred, y_proba_b, classes_b)

    assert metrics_a["pr_auc_macro"] == pytest.approx(metrics_b["pr_auc_macro"])
    assert metrics_a["brier_macro"] == pytest.approx(metrics_b["brier_macro"])
    assert metrics_a["per_class_f1"] == pytest.approx(metrics_b["per_class_f1"])


def test_format_experiments_row_matches_column_order() -> None:
    metrics = {"pr_auc_macro": 0.5123, "minority_f1": 0.4567, "brier_macro": 0.1234}

    row = format_experiments_row(
        date="2026-08-29",
        model_config="baseline_v0",
        dataset_snapshot="kubernetes-kubernetes",
        metrics=metrics,
        gate_result="n/a -- no promotion gate yet",
        notes="",
    )

    fields = [f.strip() for f in row.strip("|").split("|")]
    assert len(fields) == 8
    assert fields[0] == "2026-08-29"
    assert fields[1] == "baseline_v0"
    assert fields[2] == "kubernetes-kubernetes"
    assert fields[3] == "0.5123"
    assert fields[4] == "0.4567"
    assert fields[5] == "0.1234"
    assert fields[6] == "n/a -- no promotion gate yet"
    assert fields[7] == ""


def test_run_training_pipeline_end_to_end_writes_artifacts(tmp_path: Path) -> None:
    train_rows = make_rows(n_per_class=8)
    val_rows = make_rows(n_per_class=3, start_issue=1000)
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_rows(train_rows, train_path)
    write_rows(val_rows, val_path)

    model_out = tmp_path / "model.joblib"
    metrics_out = tmp_path / "metrics.json"

    metrics = run_training_pipeline(train_path, val_path, model_out, metrics_out)

    assert model_out.exists()
    assert metrics_out.exists()

    artifact = joblib.load(model_out)  # pyright: ignore
    assert isinstance(artifact, BaselineArtifact)
    vectorizers = FittedVectorizers(word=artifact.word_vectorizer, char=artifact.char_vectorizer)
    val_matrix = build_feature_matrix(val_rows, vectorizers)
    predictions = artifact.model.predict(val_matrix)  # pyright: ignore
    assert len(predictions) == len(val_rows)  # pyright: ignore[reportUnknownArgumentType]

    saved_metrics = json.loads(metrics_out.read_text())
    assert saved_metrics == metrics
    for key in (
        "accuracy",
        "macro_f1",
        "per_class_f1",
        "minority_f1",
        "pr_auc_macro",
        "brier_macro",
    ):
        assert key in saved_metrics


def test_format_config_label_embeds_the_actual_hyperparameters_used() -> None:
    """Guards against a static description string once hyperparameters are
    CLI-tunable -- the label must reflect whatever config produced it.
    """
    config = {
        "word_ngram_range": (1, 1),
        "word_min_df": 5,
        "word_sublinear_tf": False,
        "model_type": "logreg",
        "model_hyperparams": {"C": 0.1, "penalty": "l1"},
    }

    label = format_config_label(config)

    assert "logreg({'C': 0.1, 'penalty': 'l1'})" in label
    assert "1-1" in label
    assert "min_df=5" in label
    assert "sublinear_tf=False" in label


def test_parse_args_accepts_hyperparameter_overrides() -> None:
    args = parse_args(
        [
            "--train",
            "a/train.jsonl",
            "--val",
            "a/val.jsonl",
            "--C",
            "0.5",
            "--penalty",
            "l1",
            "--word-ngram-max",
            "1",
            "--word-min-df",
            "7",
            "--no-word-sublinear-tf",
        ]
    )

    assert pytest.approx(0.5) == args.C
    assert args.penalty == "l1"
    assert args.word_ngram_max == 1
    assert args.word_min_df == 7
    assert args.word_sublinear_tf is False


def test_run_training_pipeline_matches_train_and_evaluate_with_no_overrides(
    tmp_path: Path,
) -> None:
    """Regression guard for the run_training_pipeline/train_and_evaluate split
    (Experiment 2): the thin disk-I/O wrapper must produce byte-identical
    metrics to calling the extracted pure core directly with defaults.
    """
    train_rows = make_rows(n_per_class=8)
    val_rows = make_rows(n_per_class=3, start_issue=1000)
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_rows(train_rows, train_path)
    write_rows(val_rows, val_path)

    pipeline_metrics = run_training_pipeline(
        train_path, val_path, tmp_path / "model.joblib", tmp_path / "metrics.json"
    )
    _, core_metrics = train_and_evaluate(train_rows, val_rows)

    assert pipeline_metrics == core_metrics


def test_main_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train_rows = make_rows(n_per_class=8)
    val_rows = make_rows(n_per_class=3, start_issue=1000)
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_rows(train_rows, train_path)
    write_rows(val_rows, val_path)

    model_out = tmp_path / "model.joblib"
    metrics_out = tmp_path / "metrics.json"

    main(
        [
            "--train",
            str(train_path),
            "--val",
            str(val_path),
            "--model-out",
            str(model_out),
            "--metrics-out",
            str(metrics_out),
            "--notes",
            "test run",
        ]
    )

    assert model_out.exists()
    assert metrics_out.exists()

    out = capsys.readouterr().out
    assert '"macro_f1"' in out
    assert any(line.strip().startswith("|") for line in out.splitlines())
    assert "test run" in out


def test_main_registers_staging_candidate_with_params_and_metrics_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real local SQLite MLflow instance, no mocking, per repo convention.
    Asserts main() results in exactly one `staging` version registered, with
    the hyperparameter/library-version/row-count params and the metrics (minus
    per_class_f1, which is logged separately as a dict artifact) present on
    its run.
    """
    train_rows = make_rows(n_per_class=8)
    val_rows = make_rows(n_per_class=3, start_issue=1000)
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_rows(train_rows, train_path)
    write_rows(val_rows, val_path)

    # main() calls mlflow.set_tracking_uri(), which mutates process-global
    # state (confirmed in tests/test_registry.py) -- capture the true prior
    # state *before* pointing MLFLOW_TRACKING_URI at tmp_path, and restore it
    # afterward so this test can't leak its tmp_path db into any other test.
    original_tracking_uri = mlflow.get_tracking_uri()  # pyright: ignore[reportUnknownMemberType]

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        main(
            [
                "--train",
                str(train_path),
                "--val",
                str(val_path),
                "--model-out",
                str(tmp_path / "model.joblib"),
                "--metrics-out",
                str(tmp_path / "metrics.json"),
            ]
        )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)  # pyright: ignore[reportUnknownMemberType]

    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{DEFAULT_MODEL_NAME}'")
    assert len(versions) == 1
    version = versions[0]
    assert version.tags.get("stage") == "staging"
    assert version.run_id is not None

    run = client.get_run(version.run_id)  # pyright: ignore[reportUnknownMemberType]
    params: dict[str, str] = run.data.params  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert params["model_version"] == "baseline_v0"
    assert params["classifier"] == "logreg"
    assert params["class_weight"] == "balanced"
    assert int(params["train_row_count"]) == len(train_rows)
    assert int(params["val_row_count"]) == len(val_rows)

    run_metrics: dict[str, float] = run.data.metrics  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert "macro_f1" in run_metrics
    assert "per_class_f1" not in run_metrics

    experiment = client.get_experiment(run.info.experiment_id)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    assert experiment.name == "baseline_v0"  # pyright: ignore[reportUnknownMemberType]


def test_main_logs_hyperparameter_overrides_not_the_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Companion to the test above: when the CLI is invoked with overrides
    (the second step of Experiment 2's sweep -> re-run flow), the logged
    params must reflect those overrides, not the DEFAULT_* constants.
    """
    train_rows = make_rows(n_per_class=8)
    val_rows = make_rows(n_per_class=3, start_issue=1000)
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_rows(train_rows, train_path)
    write_rows(val_rows, val_path)

    original_tracking_uri = mlflow.get_tracking_uri()  # pyright: ignore[reportUnknownMemberType]
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        main(
            [
                "--train",
                str(train_path),
                "--val",
                str(val_path),
                "--model-out",
                str(tmp_path / "model.joblib"),
                "--metrics-out",
                str(tmp_path / "metrics.json"),
                "--C",
                "0.1",
                "--penalty",
                "l1",
                "--word-ngram-max",
                "1",
                "--word-min-df",
                "1",
                "--no-word-sublinear-tf",
            ]
        )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)  # pyright: ignore[reportUnknownMemberType]

    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{DEFAULT_MODEL_NAME}'")
    run_id = versions[0].run_id
    assert run_id is not None
    run = client.get_run(run_id)  # pyright: ignore[reportUnknownMemberType]
    params: dict[str, str] = run.data.params  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    assert params["model_hyperparams.C"] == "0.1"
    assert params["model_hyperparams.penalty"] == "l1"
    assert params["word_ngram_range"] == "1-1"
    assert params["word_min_df"] == "1"
    assert params["word_sublinear_tf"] == "False"

    artifact: BaselineArtifact = joblib.load(tmp_path / "model.joblib")  # pyright: ignore
    assert pytest.approx(1.0) == artifact.model.l1_ratio  # pyright: ignore
    assert pytest.approx(0.1) == artifact.model.C  # pyright: ignore


@pytest.mark.slow
def test_artifact_from_a_real_dash_m_invocation_is_loadable_in_another_process(
    tmp_path: Path,
) -> None:
    """Regression test for a real bug: `python -m signalscore.training.train_baseline`
    (the exact invocation this module's own docstring instructs) sets that
    module's __name__ to "__main__", so BaselineArtifact.__module__ was
    "__main__" at joblib.dump() time -- any OTHER process (the gate CLI, a
    serving process, this test) unpickling the file looked for
    BaselineArtifact on its own __main__ and raised AttributeError. Only a
    real subprocess run via `-m` reproduces this; calling main() in-process
    (as the sibling test above does) never hits it, since __main__ there is
    always pytest's own entry point.
    """
    train_rows = make_rows(n_per_class=8)
    val_rows = make_rows(n_per_class=3, start_issue=1000)
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_rows(train_rows, train_path)
    write_rows(val_rows, val_path)
    model_out = tmp_path / "model.joblib"

    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "signalscore.training.train_baseline",
            "--train",
            str(train_path),
            "--val",
            str(val_path),
            "--model-out",
            str(model_out),
            "--metrics-out",
            str(tmp_path / "metrics.json"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}"},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    artifact: BaselineArtifact = joblib.load(model_out)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    assert isinstance(artifact, BaselineArtifact)


def test_parse_args_accepts_explicit_train_and_val_paths() -> None:
    args = parse_args(["--train", "a/train.jsonl", "--val", "a/val.jsonl"])

    assert args.train == Path("a/train.jsonl")
    assert args.val == Path("a/val.jsonl")
    assert args.model_out is None
    assert args.metrics_out is None
