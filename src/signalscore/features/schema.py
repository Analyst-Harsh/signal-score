"""The assembled feature-matrix row contract.

`extra="forbid"` is the automated, structural version of the temporal-leakage
check: a dict carrying `updated_at`, `labels`, `closed_at`, or any other
unwhitelisted field raises `ValidationError` at construction instead of
silently adding a column to the training frame.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from signalscore.features.labels import Priority


class FeatureRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    issue_number: int
    created_at: datetime  # split-key only, never a model feature
    label: Priority
    text: str
    is_member_plus: bool
    title_chars: int
    body_chars: int
    body_words: int
    has_code_block: bool
    has_stack_trace: bool
    has_url: bool
    has_logline: bool
    has_excl: bool
    code_ratio: float = Field(ge=0.0, le=1.0)
    any_lex: bool
    is_ci_flake_shaped: bool
