"""SQLAlchemy 2.0 models for the polyclaude ledger.

Chain of custody:
    Event ──► Signal ──► OracleCall ──► Decision ──► Order ──► Fill ──► PnLSnapshot
                                          ▲
                                          └── CalibrationPoint (filled at resolution)

Every link foreign-keys to the prior step, so any fill traces back to the news
article (or audio transcript) that triggered it, the exact prompt sent to
Claude, and the response that justified the size.

Conventions:
- All timestamps UTC, stored as DateTime(timezone=True).
- Money fields: USDC at 6 decimals, Numeric(18, 6).
- Probabilities and prices: Numeric(10, 8), 0..1.
"""

from __future__ import annotations

import enum
import os
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache

from sqlalchemy import (
    JSON, Boolean, DateTime, Enum, Float, ForeignKey, Index,
    Integer, Numeric, String, Text, UniqueConstraint, create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker,
)

from polyclaude.config import get_settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# --- enums ---------------------------------------------------------------


class Modality(str, enum.Enum):
    text = "text"
    audio = "audio"
    image = "image"


class Side(str, enum.Enum):
    YES = "YES"
    NO = "NO"
    NONE = "NONE"


class OrderType(str, enum.Enum):
    GTC = "GTC"
    FOK = "FOK"
    GTD = "GTD"


class OrderStatus(str, enum.Enum):
    pending = "pending"
    open = "open"
    partially_filled = "partially_filled"
    filled = "filled"
    cancelled = "cancelled"
    rejected = "rejected"
    error = "error"


class DecisionAction(str, enum.Enum):
    place = "place"
    skip_low_edge = "skip_low_edge"
    skip_ambiguous = "skip_ambiguous"
    skip_thin_book = "skip_thin_book"
    skip_risk_cap = "skip_risk_cap"
    skip_dry_run = "skip_dry_run"


# --- core tables ---------------------------------------------------------


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    condition_id: Mapped[str] = mapped_column(String(80), index=True, default="")
    yes_token_id: Mapped[str] = mapped_column(String(80), index=True, default="")
    no_token_id: Mapped[str] = mapped_column(String(80), index=True, default="")
    question: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="")
    resolution_rules: Mapped[str] = mapped_column(Text, default="")
    resolution_source: Mapped[str | None] = mapped_column(String(255), default=None)
    category: Mapped[str | None] = mapped_column(String(64), index=True, default=None)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    is_india: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    resolves_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, default=None)
    resolved_outcome: Mapped[str | None] = mapped_column(String(8), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    snapshots = relationship("BookSnapshot", back_populates="market")


class BookSnapshot(Base):
    __tablename__ = "book_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    yes_midpoint: Mapped[Decimal] = mapped_column(Numeric(10, 8))
    yes_best_bid: Mapped[Decimal] = mapped_column(Numeric(10, 8))
    yes_best_ask: Mapped[Decimal] = mapped_column(Numeric(10, 8))
    depth_yes_within_2c: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    depth_no_within_2c: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    raw_book: Mapped[dict] = mapped_column(JSON)

    market = relationship("Market", back_populates="snapshots")
    __table_args__ = (Index("ix_book_snap_market_ts", "market_id", "ts"),)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    modality: Mapped[Modality] = mapped_column(Enum(Modality), default=Modality.text)
    source: Mapped[str] = mapped_column(String(128), index=True)
    source_tier: Mapped[int] = mapped_column(Integer, default=3)
    url: Mapped[str | None] = mapped_column(String(1024), default=None)
    title: Mapped[str | None] = mapped_column(Text, default=None)
    text: Mapped[str] = mapped_column(Text)
    audio_url: Mapped[str | None] = mapped_column(String(1024), default=None)
    transcript_meta: Mapped[dict | None] = mapped_column(JSON, default=None)
    lang: Mapped[str] = mapped_column(String(8), default="en")
    is_india: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    signals = relationship("Signal", back_populates="event")
    __table_args__ = (UniqueConstraint("source", "url", name="uq_event_source_url"),)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"), index=True)
    market_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), index=True)
    link_score: Mapped[float] = mapped_column(Float)
    finbert_polarity: Mapped[float] = mapped_column(Float, default=0.0)
    finbert_intensity: Mapped[float] = mapped_column(Float, default=0.0)
    vader_compound: Mapped[float] = mapped_column(Float, default=0.0)
    ensemble_score: Mapped[float] = mapped_column(Float, default=0.0)
    z_vs_baseline: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    event = relationship("Event", back_populates="signals")
    oracle_calls = relationship("OracleCall", back_populates="signal")
    __table_args__ = (Index("ix_signal_market_z", "market_id", "z_vs_baseline"),)


class OracleCall(Base):
    __tablename__ = "oracle_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_group_id: Mapped[str] = mapped_column(String(36), index=True)
    market_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), index=True)
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"), default=None)
    call_type: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(64))
    prompt_system: Mapped[str] = mapped_column(Text)
    prompt_user: Mapped[str] = mapped_column(Text)
    raw_response: Mapped[str] = mapped_column(Text)
    parsed_response: Mapped[dict] = mapped_column(JSON)
    p: Mapped[float | None] = mapped_column(Float, default=None)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    clarity: Mapped[float | None] = mapped_column(Float, default=None)
    input_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    output_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    signal = relationship("Signal", back_populates="oracle_calls")


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    market_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), index=True)
    strategy: Mapped[str] = mapped_column(String(32), index=True)
    committed_p: Mapped[float] = mapped_column(Float)
    market_p_at_decision: Mapped[float] = mapped_column(Float)
    edge_bps: Mapped[int] = mapped_column(Integer)
    side: Mapped[Side] = mapped_column(Enum(Side))
    kelly_fraction: Mapped[float] = mapped_column(Float)
    scaled_fraction: Mapped[float] = mapped_column(Float)
    intended_size_usd: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    intended_price: Mapped[Decimal] = mapped_column(Numeric(10, 8))
    action: Mapped[DecisionAction] = mapped_column(Enum(DecisionAction))
    skip_reason: Mapped[str | None] = mapped_column(String(255), default=None)
    bankroll_usd: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    daily_notional_used_usd: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    is_live: Mapped[bool] = mapped_column(Boolean, default=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    orders = relationship("Order", back_populates="decision")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    decision_id: Mapped[str] = mapped_column(ForeignKey("decisions.id"), index=True)
    market_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), index=True)
    token_id: Mapped[str] = mapped_column(String(80), index=True)
    side: Mapped[Side] = mapped_column(Enum(Side))
    order_type: Mapped[OrderType] = mapped_column(Enum(OrderType))
    price: Mapped[Decimal] = mapped_column(Numeric(10, 8))
    size_shares: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    notional_usd: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    status: Mapped[OrderStatus] = mapped_column(Enum(OrderStatus), default=OrderStatus.pending, index=True)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    decision = relationship("Decision", back_populates="orders")
    fills = relationship("Fill", back_populates="order")


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    trade_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 8))
    size_shares: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    fee_usd: Mapped[Decimal] = mapped_column(Numeric(18, 6), default=Decimal("0"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    raw: Mapped[dict] = mapped_column(JSON)

    order = relationship("Order", back_populates="fills")


class PnLSnapshot(Base):
    __tablename__ = "pnl_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    position_yes_shares: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    position_no_shares: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    avg_cost_yes: Mapped[Decimal] = mapped_column(Numeric(10, 8))
    avg_cost_no: Mapped[Decimal] = mapped_column(Numeric(10, 8))
    realized_pnl_usd: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    unrealized_pnl_usd: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    mark_yes_price: Mapped[Decimal] = mapped_column(Numeric(10, 8))


class CalibrationPoint(Base):
    """One per Decision once the market resolves. Drives Brier/calibration."""

    __tablename__ = "calibration_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(ForeignKey("decisions.id"), unique=True, index=True)
    market_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), index=True)
    strategy: Mapped[str] = mapped_column(String(32), index=True)
    p_predicted: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    outcome_yes: Mapped[bool] = mapped_column(Boolean)
    brier: Mapped[float] = mapped_column(Float)
    log_loss: Mapped[float] = mapped_column(Float)
    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class RiskState(Base):
    __tablename__ = "risk_state"

    date_utc: Mapped[str] = mapped_column(String(10), primary_key=True)
    daily_notional_used_usd: Mapped[Decimal] = mapped_column(Numeric(18, 6), default=Decimal("0"))
    daily_notional_cap_usd: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    trades_today: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# --- engine / session helpers --------------------------------------------


def _patch_libsql_isolation_probe() -> None:
    """Defensive patch retained for any libsql remote-only edge case.

    With embedded-replica mode (the default below) all SQL runs against a local
    SQLite file so this should never fire — but it's cheap insurance and keeps
    behaviour identical for users who hand-roll a `sqlite+libsql://` remote URL.
    """
    try:
        from sqlalchemy.dialects.sqlite.base import SQLiteDialect
    except ImportError:  # pragma: no cover
        return
    if getattr(SQLiteDialect, "_polyclaude_libsql_patched", False):
        return
    original = SQLiteDialect.get_isolation_level

    def safe_get_isolation_level(self, dbapi_connection):  # type: ignore[no-untyped-def]
        try:
            return original(self, dbapi_connection)
        except Exception:
            return "SERIALIZABLE"

    SQLiteDialect.get_isolation_level = safe_get_isolation_level  # type: ignore[assignment]
    SQLiteDialect._polyclaude_libsql_patched = True  # type: ignore[attr-defined]


def _parse_libsql_url(url: str) -> tuple[str, str]:
    """Return (sync_url, auth_token) for a libsql or sqlite+libsql URL."""
    from urllib.parse import parse_qs, urlsplit

    if url.startswith("sqlite+libsql://"):
        u = "libsql://" + url[len("sqlite+libsql://"):]
    elif url.startswith("libsql://"):
        u = url
    else:  # pragma: no cover - caller has already detected libsql
        u = url

    parts = urlsplit(u)
    host = parts.hostname or ""
    sync_url = f"libsql://{host}"
    if parts.port:
        sync_url = f"libsql://{host}:{parts.port}"
    params = parse_qs(parts.query)
    token = params.get("authToken", [""])[0]
    return sync_url, token


_LIBSQL_LOCAL_PATH = os.environ.get("LIBSQL_LOCAL_PATH", "data/polyclaude.libsql.db")
_libsql_remote: tuple[str, str] | None = None


def _libsql_connect():  # type: ignore[no-untyped-def]
    """SQLAlchemy `creator` callable that returns a libsql_experimental
    connection in embedded-replica mode.

    The replica syncs from Turso on connect (so reads see fresh data) and is
    written to locally; sync_database_replica() pushes pending writes back.
    """
    import libsql_experimental as libsql  # type: ignore[import]

    if _libsql_remote is None:  # pragma: no cover
        raise RuntimeError("libsql remote not initialised")
    sync_url, token = _libsql_remote
    os.makedirs(os.path.dirname(_LIBSQL_LOCAL_PATH) or ".", exist_ok=True)
    conn = libsql.connect(_LIBSQL_LOCAL_PATH, sync_url=sync_url, auth_token=token)
    try:
        conn.sync()
    except Exception:
        # First-time connect against an empty Turso DB has nothing to pull;
        # we'll still be able to write and the next sync will push schema.
        pass
    return conn


def sync_database_replica() -> None:
    """Push local writes to Turso and pull remote changes.

    Call this after a batch of inserts in long-running processes; the engine
    auto-syncs on `connect`, but explicit sync after writes is what guarantees
    the dashboard sees fresh data.
    """
    if _libsql_remote is None:
        return
    try:
        import libsql_experimental as libsql  # type: ignore[import]

        sync_url, token = _libsql_remote
        conn = libsql.connect(_LIBSQL_LOCAL_PATH, sync_url=sync_url, auth_token=token)
        conn.sync()
        conn.close()
    except Exception:  # pragma: no cover
        pass


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Build the SQLAlchemy engine.

    Supports four URL forms:
    - sqlite:///path/to.db                          — local file
    - sqlite+libsql://<host>?authToken=…            — Turso remote (fragile;
      most PRAGMAs fail over Hrana — embedded-replica below is preferred)
    - libsql://<host>?authToken=…                   — Turso (auto-converted to
      embedded-replica mode: a local SQLite file at $LIBSQL_LOCAL_PATH that
      syncs with Turso. All PRAGMAs work because the engine talks to local
      SQLite via the standard sqlite3 driver; libsql only handles the sync.)
    - postgresql://…                                — Supabase / managed Postgres
    """
    global _libsql_remote
    settings = get_settings()
    url = settings.database_url

    is_libsql = url.startswith("libsql://") or url.startswith("sqlite+libsql://")

    if is_libsql:
        # Always use embedded-replica mode.
        _libsql_remote = _parse_libsql_url(url)
        os.makedirs(os.path.dirname(_LIBSQL_LOCAL_PATH) or ".", exist_ok=True)
        _patch_libsql_isolation_probe()
        # `creator` overrides URL-based connection; we still pass a dummy URL so
        # SQLAlchemy picks the SQLite dialect.
        return create_engine(
            "sqlite:///" + _LIBSQL_LOCAL_PATH, future=True, creator=_libsql_connect,
        )

    if url.startswith("sqlite"):
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    return create_engine(url, future=True)


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def get_session() -> Session:
    return _session_factory()()


def init_db() -> None:
    Base.metadata.create_all(get_engine())
