"""Anthropic-API-backed oracle.

Three call types: probability, ambiguity, sizing.

Each call returns a strict-JSON response validated by a Pydantic model. On
malformed output we retry once, then return None and let the caller skip the
market.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from polyclaude.config import Settings, get_settings
from polyclaude.logging_setup import get_logger
from polyclaude.oracle import prompts as P

log = get_logger(__name__)


# --- typed responses --------------------------------------------------------


class ProbabilityResponse(BaseModel):
    p: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    key_factors: list[str] = Field(default_factory=list, max_length=10)
    rationale: str
    reference_class: str = ""

    @field_validator("p", "confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class AmbiguityResponse(BaseModel):
    clarity: float = Field(ge=0, le=1)
    edge_cases: list[str] = Field(default_factory=list, max_length=10)
    resolution_source_quality: float = Field(ge=0, le=1)
    verdict: Literal["clear", "ambiguous", "hostile"]


class SizingResponse(BaseModel):
    would_trade: bool
    side: Literal["YES", "NO", "NONE"]
    edge_bps: int
    kelly_fraction: float
    reasoning: str = ""


@dataclass
class CallTrace:
    """Everything we need to log to OracleCall."""

    decision_group_id: str
    call_type: str
    model: str
    prompt_system: str
    prompt_user: str
    raw_response: str
    parsed: dict
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    error: str | None


# --- JSON extraction --------------------------------------------------------


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(s: str) -> dict | None:
    """Best-effort JSON extraction. Tries straight load, then largest {...} block."""
    s = s.strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    # Strip markdown code fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    m = _JSON_RE.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# --- the oracle -------------------------------------------------------------


class ClaudeOracle:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("`anthropic` package not installed.") from e
            self._client = Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def _call(
        self,
        system: str,
        user: str,
        decision_group_id: str,
        call_type: str,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> tuple[str, dict, CallTrace]:
        model = model or self.settings.anthropic_model
        t0 = time.perf_counter()
        err: str | None = None
        raw = ""
        in_tok: int | None = None
        out_tok: int | None = None
        try:
            msg = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parts = []
            for c in msg.content:
                if hasattr(c, "text"):
                    parts.append(c.text)
            raw = "".join(parts)
            usage = getattr(msg, "usage", None)
            if usage:
                in_tok = getattr(usage, "input_tokens", None)
                out_tok = getattr(usage, "output_tokens", None)
        except Exception as e:
            err = str(e)
            log.error("oracle.call_failed", call=call_type, err=err)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        parsed = extract_json(raw) or {}
        trace = CallTrace(
            decision_group_id=decision_group_id,
            call_type=call_type,
            model=model,
            prompt_system=system,
            prompt_user=user,
            raw_response=raw,
            parsed=parsed,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            error=err,
        )
        return raw, parsed, trace

    # --- public API -------------------------------------------------------

    def evaluate_ambiguity(
        self,
        market: dict,
        decision_group_id: str | None = None,
    ) -> tuple[AmbiguityResponse | None, CallTrace]:
        gid = decision_group_id or str(uuid.uuid4())
        user = P.ambiguity_user(market)
        for attempt in range(2):
            _, parsed, trace = self._call(
                P.AMBIGUITY_SYSTEM, user, gid, "ambiguity",
                model=self.settings.anthropic_model_fast, max_tokens=512,
            )
            try:
                return AmbiguityResponse.model_validate(parsed), trace
            except ValidationError as e:
                log.warning("oracle.ambiguity_parse_fail", attempt=attempt, err=str(e))
        return None, trace

    def evaluate_probability(
        self,
        market: dict,
        evidence: list[dict],
        decision_group_id: str | None = None,
    ) -> tuple[ProbabilityResponse | None, CallTrace]:
        gid = decision_group_id or str(uuid.uuid4())
        # Strip any leakage of market price from the market dict before the call.
        sanitized = {k: v for k, v in market.items() if k not in ("yes_price", "market_p", "midpoint")}
        hint = P.hint_for(sanitized.get("category"), sanitized.get("tags"))
        user = P.probability_user(sanitized, evidence, hint)
        for attempt in range(2):
            _, parsed, trace = self._call(P.PROBABILITY_SYSTEM, user, gid, "probability")
            try:
                return ProbabilityResponse.model_validate(parsed), trace
            except ValidationError as e:
                log.warning("oracle.prob_parse_fail", attempt=attempt, err=str(e))
        return None, trace

    def evaluate_sizing(
        self,
        market: dict,
        committed_p: float,
        market_yes_price: float,
        book_depth_yes: float,
        book_depth_no: float,
        signal_volatility: float,
        bankroll_usd: float,
        decision_group_id: str | None = None,
    ) -> tuple[SizingResponse | None, CallTrace]:
        gid = decision_group_id or str(uuid.uuid4())
        user = P.sizing_user(
            market, committed_p, market_yes_price,
            book_depth_yes, book_depth_no, signal_volatility, bankroll_usd,
        )
        for attempt in range(2):
            _, parsed, trace = self._call(
                P.SIZING_SYSTEM, user, gid, "sizing",
                model=self.settings.anthropic_model_fast, max_tokens=512,
            )
            try:
                return SizingResponse.model_validate(parsed), trace
            except ValidationError as e:
                log.warning("oracle.sizing_parse_fail", attempt=attempt, err=str(e))
        return None, trace
