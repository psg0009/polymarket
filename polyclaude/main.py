"""CLI entry point + async scheduler.

Sub-commands:
    polyclaude scan        # one-shot dry-run scan, print decision table, write ledger
    polyclaude trade       # live trading loop (requires --live + confirmation)
    polyclaude reconcile   # resolve fills against CLOB, fill CalibrationPoints
    polyclaude backtest    # walk-forward replay (delegates to backtest.harness)
    polyclaude preflight   # print allowance preflight only
    polyclaude init        # initialise the SQLite schema
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from polyclaude.clob.allowances import preflight
from polyclaude.clob.client import ClobWrapper
from polyclaude.clob.executor import Executor, PlaceRequest
from polyclaude.config import get_settings
from polyclaude.ledger.db import init_db
from polyclaude.ledger.persist import (
    get_or_create_risk_state, increment_daily_used, save_decision,
    save_oracle_calls, save_order, upsert_market,
)
from polyclaude.ledger.reconcile import (
    daily_brier_summary, fill_calibration_points, reconcile_fills,
)
from polyclaude.logging_setup import configure as configure_logging
from polyclaude.logging_setup import new_correlation_id
from polyclaude.markets.gamma import GammaClient, MarketSummary
from polyclaude.markets.india import is_india_market
from polyclaude.oracle.claude import ClaudeOracle
from polyclaude.strategy.value import ValueStrategy

console = Console()


def _market_to_dict(m: MarketSummary) -> dict[str, Any]:
    return {
        "id": m.id,
        "question": m.question,
        "description": m.description,
        "resolution_rules": m.resolution_rules,
        "resolution_source": m.resolution_source,
        "resolves_at": m.end_date.isoformat() if m.end_date else "unknown",
        "category": m.category,
        "tags": m.tags,
        "hours_to_resolution": m.hours_to_resolution,
    }


async def _scan_once(*, limit: int, india_only: bool, is_live: bool, max_notional: Decimal) -> None:
    settings = get_settings()
    init_db()
    new_correlation_id()

    rs = get_or_create_risk_state(min(settings.max_notional_per_day, max_notional))
    bankroll = Decimal(str(max_notional))
    daily_used = rs.daily_notional_used_usd or Decimal("0")

    clob = ClobWrapper(settings)
    oracle = ClaudeOracle(settings)
    strat = ValueStrategy(oracle, settings)
    executor = Executor(clob, settings)

    table = Table(title=f"polyclaude scan ({'LIVE' if is_live else 'DRY-RUN'})")
    for col in ("market", "p", "mkt", "edge¢", "side", "size$", "action", "reason"):
        table.add_column(col, no_wrap=col != "market")

    async with GammaClient(settings) as gamma:
        markets = await gamma.list_markets(limit=limit)

    if india_only:
        markets = [m for m in markets if is_india_market(m)]

    for m in markets:
        if not m.yes_token_id or m.closed:
            continue
        upsert_market(m)
        try:
            yes_price = clob.get_midpoint(m.yes_token_id) or (m.yes_price or 0.5)
        except Exception:
            yes_price = m.yes_price or 0.5
        try:
            book = clob.get_orderbook(m.yes_token_id)
            from polyclaude.markets.snapshots import parse_book

            stats = parse_book(book)
            depth_yes = Decimal(str(stats["depth_yes_within_2c"]))
            depth_no = Decimal(str(stats["depth_no_within_2c"]))
        except Exception:
            depth_yes = Decimal("0")
            depth_no = Decimal("0")

        market_dict = _market_to_dict(m)
        if not settings.has_anthropic():
            console.log(f"[yellow]ANTHROPIC_API_KEY not set; skipping oracle for {m.id[:8]}…[/yellow]")
            continue

        decision, prob, traces = strat.decide(
            market=market_dict, evidence=[], market_yes_price=float(yes_price),
            book_depth_yes_usd=depth_yes, book_depth_no_usd=depth_no,
            bankroll_usd=bankroll, daily_used_usd=daily_used,
        )
        save_oracle_calls(traces, m.id)
        save_decision(decision, bankroll_usd=bankroll, daily_used_usd=daily_used, is_live=is_live)

        sz = decision.sizing
        table.add_row(
            (m.question or "")[:60],
            f"{decision.committed_p:.3f}",
            f"{yes_price:.3f}",
            f"{sz.edge_bps/100:+.1f}",
            sz.side,
            f"{float(sz.notional_usd):.2f}",
            decision.action.value,
            (decision.skip_reason or "")[:50],
        )

        if decision.action.name == "place" and sz.notional_usd > 0:
            token = m.yes_token_id if sz.side == "YES" else m.no_token_id
            shares = (sz.notional_usd / sz.price).quantize(Decimal("0.01"))
            req = PlaceRequest(
                decision_id=decision.decision_group_id,
                market_id=m.id, token_id=token, side=sz.side,
                price=sz.price, size_shares=shares,
            )
            res = executor.place(req)
            save_order(
                decision_id=decision.decision_group_id, market_id=m.id, token_id=token,
                side=sz.side, price=sz.price, size_shares=shares,
                notional=sz.notional_usd, result=res,
            )
            if res.ok:
                daily_used = increment_daily_used(sz.notional_usd)

    console.print(table)
    console.print(f"[dim]daily notional used: ${float(daily_used):.2f}[/dim]")


# --- click CLI -----------------------------------------------------------


@click.group()
@click.option("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
def cli(log_level: str | None) -> None:
    settings = get_settings()
    configure_logging(log_level or settings.log_level)


@cli.command("init")
def cmd_init() -> None:
    """Create the SQLite schema."""
    init_db()
    console.print("[green]Database initialized.[/green]")


@cli.command("preflight")
def cmd_preflight() -> None:
    """Print allowance preflight without trading."""
    res = preflight()
    console.print(res.explain())


@cli.command("scan")
@click.option("--limit", default=50, help="how many markets to scan from Gamma")
@click.option("--india-only", is_flag=True, default=False)
@click.option("--max-notional", default=None, help="override per-day notional cap (USD)")
def cmd_scan(limit: int, india_only: bool, max_notional: str | None) -> None:
    """One-shot dry-run scan."""
    s = get_settings()
    cap = Decimal(max_notional) if max_notional else s.max_notional_per_day
    asyncio.run(_scan_once(limit=limit, india_only=india_only or s.india_only,
                           is_live=False, max_notional=cap))


@cli.command("trade")
@click.option("--limit", default=50)
@click.option("--india-only", is_flag=True, default=False)
@click.option("--max-notional", required=True)
@click.option("--live/--no-live", default=False)
@click.option("--yes-i-am-sure", is_flag=True, default=False, help="bypass confirmation prompt")
def cmd_trade(limit: int, india_only: bool, max_notional: str, live: bool, yes_i_am_sure: bool) -> None:
    """Live trading loop."""
    s = get_settings()
    cap = Decimal(max_notional)

    if live:
        if cap > Decimal("500"):
            raise click.UsageError("--max-notional > $500 requires editing the source as a safety check.")
        # Hard preflight gate
        pre = preflight(s)
        if not pre.ok:
            console.print(pre.explain())
            raise click.UsageError("Allowance preflight failed; refusing to trade.")
        if not yes_i_am_sure:
            confirm = click.prompt(
                f"Type the cap exactly to confirm live trading (max ${cap})", type=str
            )
            if confirm.strip() != str(cap):
                raise click.UsageError("Confirmation did not match.")
    asyncio.run(_scan_once(limit=limit, india_only=india_only or s.india_only,
                           is_live=live, max_notional=cap))


@cli.command("reconcile")
def cmd_reconcile() -> None:
    """Reconcile open orders against client.get_trades(); fill CalibrationPoints."""
    init_db()
    clob = ClobWrapper(get_settings())
    inserted = reconcile_fills(clob)
    created = fill_calibration_points()
    summary = daily_brier_summary()
    console.print(f"[green]fills inserted={inserted} calibration points created={created}[/green]")
    console.print(summary)


@cli.command("backtest")
@click.option("--from", "from_", required=True, help="ISO date, e.g. 2025-01-01")
@click.option("--to", "to", required=True, help="ISO date")
@click.option("--strategy", default="value")
@click.option("--capital", default="1000")
def cmd_backtest(from_: str, to: str, strategy: str, capital: str) -> None:
    """Walk-forward replay of historical snapshots through the live pipeline."""
    from polyclaude.backtest.harness import run as run_backtest

    start = datetime.fromisoformat(from_).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(to).replace(tzinfo=timezone.utc)
    metrics = run_backtest(start=start, end=end, strategy=strategy, capital=Decimal(capital))
    console.print(metrics)


if __name__ == "__main__":
    cli()
