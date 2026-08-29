"""The 4 leakage-audit checks run before a split is DVC-frozen.

Check 3 (temporal leakage) is not a runtime function here — it's structural,
verified by tests/features/test_pipeline.py: `FeatureRow(extra="forbid")`
rejects any unwhitelisted key, and `build_features()`'s 3-arg signature has no
scope to read banned fields.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import (
    TfidfVectorizer,  # pyright: ignore[reportMissingTypeStubs]
)
from sklearn.metrics.pairwise import cosine_similarity  # pyright: ignore[reportUnknownVariableType]

from signalscore.features.labels import LabelMappingConfig, Priority
from signalscore.features.schema import FeatureRow


def _pairwise_cosine_similarity(texts: Sequence[str]) -> NDArray[np.float64]:
    """Isolates sklearn's untyped surface to one place — sklearn ships no py.typed
    marker, so its return types are `Unknown` under pyright strict; every caller
    of this helper works with a fully-typed numpy array instead.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)  # pyright: ignore
    result = cosine_similarity(matrix)  # pyright: ignore
    return result  # pyright: ignore[reportUnknownVariableType]


@dataclass
class NearDuplicatePair:
    issue_a: int
    issue_b: int
    split_a: str
    split_b: str
    cosine_similarity: float
    cross_split: bool


def find_near_duplicates(
    rows: Sequence[FeatureRow], split_of: Mapping[int, str], threshold: float = 0.85
) -> list[NearDuplicatePair]:
    """Flag near-duplicate issue text for manual review — never auto-dropped.

    Fits a throwaway TF-IDF vectorizer across the entire pool purely for
    similarity scoring (it is never shipped, so fitting across all splits is
    not a leakage risk to the model itself). Runs after split assignment: the
    useful signal is pairs that straddle a split boundary.

    # ponytail: naive dense cosine_similarity(X) at n~11,899 is ~1GB and
    # borderline; batch in row chunks if memory becomes an issue. threshold
    # is a starting point, calibrate against the top ~50 real pairs once run.
    """
    if len(rows) < 2:
        return []
    similarity = _pairwise_cosine_similarity([row.text for row in rows])

    pairs: list[NearDuplicatePair] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            score = float(similarity[i, j])
            if score >= threshold:
                row_a, row_b = rows[i], rows[j]
                split_a = split_of.get(row_a.issue_number, "unknown")
                split_b = split_of.get(row_b.issue_number, "unknown")
                pairs.append(
                    NearDuplicatePair(
                        issue_a=row_a.issue_number,
                        issue_b=row_b.issue_number,
                        split_a=split_a,
                        split_b=split_b,
                        cosine_similarity=score,
                        cross_split=split_a != split_b,
                    )
                )
    pairs.sort(key=lambda pair: pair.cosine_similarity, reverse=True)
    return pairs


@dataclass
class AuthorConcentrationReport:
    split: str
    n_rows: int
    top1_author_share: float
    top10_author_share: float
    hhi: float
    flagged: bool


def audit_author_concentration(
    author_by_number: Mapping[int, str],
    split_of: Mapping[int, str],
    hhi_threshold: float = 0.02,
    top1_threshold: float = 0.05,
) -> list[AuthorConcentrationReport]:
    """Flags (does not block) a split dominated by a small number of authors.

    Thresholds are placeholders — calibrate against this corpus's real
    login-level distribution once computed, don't ship as final.
    """
    numbers_by_split: dict[str, list[int]] = {}
    for number, split in split_of.items():
        numbers_by_split.setdefault(split, []).append(number)

    reports: list[AuthorConcentrationReport] = []
    for split, numbers in numbers_by_split.items():
        authors = [author_by_number[n] for n in numbers if n in author_by_number]
        n_rows = len(authors)
        if n_rows == 0:
            continue
        counts = sorted(Counter(authors).values(), reverse=True)
        top1_share = counts[0] / n_rows
        top10_share = sum(counts[:10]) / n_rows
        hhi = sum((count / n_rows) ** 2 for count in counts)
        reports.append(
            AuthorConcentrationReport(
                split=split,
                n_rows=n_rows,
                top1_author_share=top1_share,
                top10_author_share=top10_share,
                hhi=hhi,
                flagged=hhi > hhi_threshold or top1_share > top1_threshold,
            )
        )
    return reports


@dataclass
class LabelDriftRow:
    issue_number: int
    queried_labels: set[str]
    resolved_label: Priority
    drifted: bool


def audit_label_drift(
    kept_rows: Sequence[dict[str, Any]], repo: str, config: LabelMappingConfig
) -> list[LabelDriftRow]:
    """Recompute each queried label's mapped Priority and compare it to the priority
    resolved from the issue's current live labels. Returns only the drifted rows —
    an ongoing, re-verified guarantee rather than an assumption that stays true
    forever once a corpus is refetched.
    """
    repo_config = config.repos[repo]
    drifted: list[LabelDriftRow] = []
    for row in kept_rows:
        resolved: Priority = row["_priority"]
        queried_labels: set[str] = row["_queried_labels"]
        queried_priorities = {repo_config.mapping.get(label) for label in queried_labels}
        if queried_priorities != {resolved}:
            drifted.append(
                LabelDriftRow(
                    issue_number=row["number"],
                    queried_labels=queried_labels,
                    resolved_label=resolved,
                    drifted=True,
                )
            )
    return drifted
