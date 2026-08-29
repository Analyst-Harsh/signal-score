"""Tests for signalscore.features.text."""

from signalscore.features.text import extract_text_features


def test_extract_text_features_builds_combined_text_from_title_and_body() -> None:
    result = extract_text_features("pod crashes", "it just dies")

    assert result["text"] == "pod crashes\nit just dies"


def test_extract_text_features_truncates_body_to_2000_chars() -> None:
    long_body = "x" * 3000

    result = extract_text_features("title", long_body)

    # +1 for the title's own char, +1 for the newline separator.
    assert len(result["text"]) == len("title") + 1 + 2000


def test_extract_text_features_normalizes_digits_to_hash_symbol() -> None:
    result = extract_text_features("broken in 2017", "")

    assert result["text"] == "broken in ####"


def test_extract_text_features_normalizes_long_hex_runs_to_hash_token() -> None:
    result = extract_text_features("build abcdef1234567 failed", "")

    assert "HASH" in result["text"]
    assert "abcdef1234567" not in result["text"]


def test_extract_text_features_drops_month_names() -> None:
    result = extract_text_features("failure on sep release", "")

    assert "sep" not in result["text"].lower()


def test_extract_text_features_strips_html_comments() -> None:
    result = extract_text_features("title", "before <!-- hidden template junk --> after")

    assert "hidden template junk" not in result["text"]
    assert "before" in result["text"]
    assert "after" in result["text"]


def test_extract_text_features_strips_prow_command_lines() -> None:
    result = extract_text_features("title", "real content\n/assign someone\nmore content")

    assert "/assign" not in result["text"]
    assert "real content" in result["text"]
    assert "more content" in result["text"]


def test_extract_text_features_strips_template_headers_keeps_prose() -> None:
    body = "**What happened**\nthe pod crashed on startup"

    result = extract_text_features("title", body)

    assert "What happened" not in result["text"]
    assert "the pod crashed on startup" in result["text"]


def test_extract_text_features_has_code_block_true_for_fenced_block() -> None:
    with_block = extract_text_features("title", "```\ncode here\n```")
    without_block = extract_text_features("title", "no code here")

    assert with_block["has_code_block"] is True
    assert without_block["has_code_block"] is False


def test_extract_text_features_has_stack_trace_true_for_goroutine() -> None:
    with_trace = extract_text_features("title", "panic: goroutine 5 [running]")
    without_trace = extract_text_features("title", "just a normal report")

    assert with_trace["has_stack_trace"] is True
    assert without_trace["has_stack_trace"] is False


def test_extract_text_features_has_url_true_for_http_link() -> None:
    with_url = extract_text_features("title", "see https://example.com/log for details")
    without_url = extract_text_features("title", "no links here")

    assert with_url["has_url"] is True
    assert without_url["has_url"] is False


def test_extract_text_features_has_logline_true_for_klog_prefix() -> None:
    with_logline = extract_text_features("title", "E0921 14:22:01.123456 apiserver error")
    without_logline = extract_text_features("title", "plain english description")

    assert with_logline["has_logline"] is True
    assert without_logline["has_logline"] is False


def test_extract_text_features_has_excl_true_for_exclamation() -> None:
    with_excl = extract_text_features("this is broken!", "")
    without_excl = extract_text_features("this is broken", "")

    assert with_excl["has_excl"] is True
    assert without_excl["has_excl"] is False


def test_extract_text_features_code_ratio_computed_correctly() -> None:
    body = "abc```code```"  # 3 body chars outside the fence, 4 inside ("code")

    result = extract_text_features("title", body)

    assert result["code_ratio"] == 4 / len(body)


def test_extract_text_features_code_ratio_zero_for_empty_body() -> None:
    result = extract_text_features("title", "")

    assert result["code_ratio"] == 0.0


def test_extract_text_features_any_lex_true_for_lexicon_term() -> None:
    with_term = extract_text_features("title", "this caused a panic in the kernel")
    without_term = extract_text_features("title", "everything is working fine")

    assert with_term["any_lex"] is True
    assert without_term["any_lex"] is False


def test_extract_text_features_is_ci_flake_shaped_true_for_ci_title() -> None:
    ci_shaped = extract_text_features("e2e test flake in kubemark", "")
    not_ci_shaped = extract_text_features("kubelet crashes on startup", "")

    assert ci_shaped["is_ci_flake_shaped"] is True
    assert not_ci_shaped["is_ci_flake_shaped"] is False


def test_extract_text_features_diagnostics_use_raw_pre_normalization_text() -> None:
    result = extract_text_features("title", "year 2017")

    # body_chars must reflect the raw, un-normalized body length, not the
    # digit-normalized text channel.
    assert result["body_chars"] == len("year 2017")
    assert result["body_words"] == 2
