"""Text cleaning, digit normalization, and diagnostic-only signal extraction.

`text` is the TF-IDF input channel; everything else in `TextFeatures` is
computed for drift monitoring and stratification only — `v0-feature-selection.md`
measured feeding these to the model as a -2.75 macro-F1 loss (95% CI
[-4.38, -1.10]), so they must never reach the model matrix.
"""

import re
from typing import TypedDict

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_PROW_COMMAND_LINE = re.compile(r"^/[a-z][\w-]*.*$", re.MULTILINE)
_TEMPLATE_HEADERS = re.compile(
    r"^\s*#{0,3}\s*\**\s*"
    r"(what happened|what you expected to happen|how to reproduce it"
    r"|anything else we need to know|environment)"
    r"\s*\**\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_HEX_RUN = re.compile(r"[0-9a-fA-F]{7,}")
_DIGIT = re.compile(r"\d")
_MONTH_NAME = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)

_CODE_FENCE = re.compile(r"```(.*?)```", re.DOTALL)
_URL = re.compile(r"https?://")
_LOGLINE = re.compile(r"^[IWEF]\d{4}\s", re.MULTILINE)
_STACK_TRACE = re.compile(r"goroutine|traceback \(most recent call last\)|\.go:\d+", re.IGNORECASE)
_CI_FLAKE_SHAPED_TITLE = re.compile(
    r"broken test run|\[k8s\.io\]|e2e|kubemark|gke-|flake", re.IGNORECASE
)
_LEXICON = (
    "panic",
    "crash",
    "crashes",
    "regression",
    "deadlock",
    "cve",
    "data loss",
    "corrupt",
    "security",
    "outage",
    "hang",
    "hangs",
    "leak",
    "oom",
    "dos",
    "broken",
    "goroutine",
    "traceback",
    "vulnerab",
    "oomkill",
    "sigsegv",
    "segfault",
)

BODY_TRUNCATE_CHARS = 2000


class TextFeatures(TypedDict):
    text: str
    title_chars: int
    body_chars: int
    body_words: int
    has_code_block: bool
    has_stack_trace: bool
    has_url: bool
    has_logline: bool
    has_excl: bool
    code_ratio: float
    any_lex: bool
    is_ci_flake_shaped: bool


def _normalize_digits(text: str) -> str:
    text = _HEX_RUN.sub("HASH", text)
    text = _DIGIT.sub("#", text)
    return _MONTH_NAME.sub("", text)


def _code_ratio(body: str) -> float:
    if not body:
        return 0.0
    code_chars = sum(len(match) for match in _CODE_FENCE.findall(body))
    return code_chars / len(body)


def extract_text_features(title: str, body: str) -> TextFeatures:
    """Build the cleaned TF-IDF text channel plus diagnostic-only columns.

    Diagnostics are computed on the raw, pre-truncation, pre-normalization
    title/body so they reflect the real issue shape rather than a truncated or
    normalized view.
    """
    combined = title.strip() + "\n" + body.strip()[:BODY_TRUNCATE_CHARS]
    combined = _HTML_COMMENT.sub("", combined)
    combined = _PROW_COMMAND_LINE.sub("", combined)
    combined = _TEMPLATE_HEADERS.sub("", combined)
    combined = _normalize_digits(combined).strip()

    raw_combined = f"{title}\n{body}"
    return {
        "text": combined,
        "title_chars": len(title),
        "body_chars": len(body),
        "body_words": len(body.split()),
        "has_code_block": "```" in body,
        "has_stack_trace": bool(_STACK_TRACE.search(raw_combined)),
        "has_url": bool(_URL.search(raw_combined)),
        "has_logline": bool(_LOGLINE.search(raw_combined)),
        "has_excl": "!" in raw_combined,
        "code_ratio": _code_ratio(body),
        "any_lex": any(term in raw_combined.lower() for term in _LEXICON),
        "is_ci_flake_shaped": bool(_CI_FLAKE_SHAPED_TITLE.search(title)),
    }
