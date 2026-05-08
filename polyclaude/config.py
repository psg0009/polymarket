"""Runtime configuration loaded from environment / .env."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Polymarket ---
    private_key: str = Field(default="")
    funder: str = Field(default="")
    signature_type: int = Field(default=1)
    clob_host: str = Field(default="https://clob.polymarket.com")
    gamma_host: str = Field(default="https://gamma-api.polymarket.com")
    chain_id: int = Field(default=137)

    # --- Anthropic ---
    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="claude-opus-4-7")
    anthropic_model_fast: str = Field(default="claude-haiku-4-5-20251001")

    # --- Risk caps ---
    max_notional_per_trade: Decimal = Field(default=Decimal("25"))
    max_notional_per_day: Decimal = Field(default=Decimal("100"))
    kelly_fraction: float = Field(default=0.25)
    min_edge_bps: int = Field(default=300)
    min_book_depth_usd: Decimal = Field(default=Decimal("50"))
    min_ambiguity_clarity: float = Field(default=0.7)

    # --- Mode flags ---
    dry_run: bool = Field(default=True)
    allow_aggressive: bool = Field(default=False)
    india_only: bool = Field(default=False)

    # --- Storage ---
    database_url: str = Field(default="sqlite:///./data/polyclaude.sqlite")
    log_level: str = Field(default="INFO")

    # --- Ingest ---
    newsapi_key: str = Field(default="")
    rss_feeds: str = Field(default="")
    india_rss_feeds: str = Field(default="")

    @field_validator("private_key")
    @classmethod
    def _strip_0x(cls, v: str) -> str:
        return v[2:] if v.startswith("0x") else v

    @property
    def rss_feeds_list(self) -> list[str]:
        return [s.strip() for s in self.rss_feeds.split(",") if s.strip()]

    @property
    def india_rss_feeds_list(self) -> list[str]:
        return [s.strip() for s in self.india_rss_feeds.split(",") if s.strip()]

    def has_trading_creds(self) -> bool:
        return bool(self.private_key) and bool(self.funder)

    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
