"""CLI entry point + async scheduler.

Sub-commands:
    polyclaude init       # initialise the SQLite schema
    polyclaude preflight  # print allowance preflight only
    polyclaude scan       # one-shot dry-run scan, print decision table, write ledger
    polyclaude trade      # one-shot live trading (requires --live + confirmation)
    polyclaude run        # long-running daemon (3 cadences: rescan / news-event / snapshots)
    polyclaude reconcile  # pull resolved outcomes, fill CalibrationPoints, reconcile fills
    polyclaude backtest   # walk-forward replay (delegates to backtest.harness)

The daemon (`run`) is the production entry point. `scan` is for one-off
inspection runs.
"""

from __future__ import annotations

import asyncio
import signal
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
from polyclaude.ledger.db import init_db, sync_database_replica
from polyclaude.ledger.persist import (
    get_or_create_risk_state, increment_daily_used, save_decision,
    save_oracle_calls, save_order, save_snapshot, upsert_market,
)
from polyclaude.ledger.reconcile import (
    daily_brier_summary, fill_calibration_points, pull_resolved_outcomes,
    reconcile_fills,
)
from polyclaude.logging_setup import configure as configure_logging
from polyclaude.logging_setup import get_logger, new_correlation_id
from polyclaude.markets.gamma import GammaClient, MarketSummary
from polyclaude.markets.india import is_india_market, vertical_for
from polyclaude.markets.snapshots import parse_book, snapshot_market
from polyclaude.oracle.claude import ClaudeOracle
from polyclaude.pipeline.discover import (
    EvidenceRow, evidence_for_oracle, hot_markets, run_discover,
)
from polyclaude.strategy.india_elections import IndiaElectionStrategy
from polyclaude.strategy.news_event import NewsEventStrategy
from polyclaude.strategy.value import ValueStrategy

console = Console()
log = get_logger(__name__)


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


def _pick_strategy(m: MarketSummary, value: ValueStrategy):
    """Route by India vertical → IndiaElectionStrategy when relevant."""
    if is_india_market(m) and (vertical_for(m) or "").startswith("india_election"):
        return IndiaElectionStrategy(inner=value)
    return value


async def _scan_pass(
    *,
    settings,
    clob: ClobWrapper,
    oracle: ClaudeOracle,
    value_strat: ValueStrategy,
    executor: Executor,
    limit: int,
    india_only: bool,
    is_live: bool,
    bankroll: Decimal,
    use_news: bool,
    only_market_ids: set[str] | None = None,
    print_table: bool = True,
) -> dict[str, list[EvidenceRow]]:
    """One pass over the top markets. Returns evidence-by-market for hot-path follow-up."""
    new_correlation_id()
    rs = get_or_create_risk_state(min(settings.max_notional_per_day, bankroll))
    daily_used = rs.daily_notional_used_usd or Decimal("0")

    async with GammaClient(settings) as gamma:
        markets = await gamma.list_markets(limit=limit)
    if india_only:
        markets = [m for m in markets if is_india_market(m)]
    if only_market_ids:
        markets = [m for m in markets if m.id in only_market_ids]

    # Persist markets BEFORE the discovery pipeline. The pipeline writes
    # Signals with market_id foreign keys; without these rows the inserts
    # crash with `FOREIGN KEY constraint failed`.
    for m in markets:
        upsert_market(m)

    # Run discovery (RSS → score → link → persist)
    by_market: dict[str, list[EvidenceRow]] = {}
    if use_news:
        try:
            by_market = await run_discover(markets, settings)
        except Exception as e:  # pragma: no cover
            log.warning("scan.discover_failed", err=str(e))

    table = Table(title=f"polyclaude scan ({'LIVE' if is_live else 'DRY-RUN'})")
    for col in ("market", "p", "mkt", "edge¢", "side", "size$", "evid", "action", "reason"):
        table.add_column(col, no_wrap=col != "market")

    for m in markets:
        if not m.yes_token_id or m.closed:
            continue
        # Market was upserted above before discover; no need to re-upsert.
        # Snapshot (P0.2) — book + depth + raw, persisted for backtest fuel.
        try:
            snap = snapshot_market(clob, m.id, m.yes_token_id)
            save_snapshot(snap)
            if snap is not None:
                yes_price = float(snap.yes_midpoint) or (m.yes_price or 0.5)
                depth_yes = Decimal(snap.depth_yes_within_2c)
                depth_no = Decimal(snap.depth_no_within_2c)
            else:
                raise RuntimeError("no snapshot")
        except Exception:
            try:
                yes_price = clob.get_midpoint(m.yes_token_id) or (m.yes_price or 0.5)
                book = clob.get_orderbook(m.yes_token_id)
                stats = parse_book(book)
                depth_yes = Decimal(str(stats["depth_yes_within_2c"]))
                depth_no = Decimal(str(stats["depth_no_within_2c"]))
            except Exception:
                yes_price = m.yes_price or 0.5
                depth_yes = Decimal("0")
                depth_no = Decimal("0")

        if not settings.has_anthropic():
            console.log(
                f"[yellow]ANTHROPIC_API_KEY not set; skipping oracle for {m.id[:8]}…[/yellow]"
            )
            continue

        market_dict = _market_to_dict(m)
        evidence_rows = by_market.get(m.id, [])
        evidence = evidence_for_oracle(evidence_rows, top_k=15)
        signal_z = max((abs(r.z) for r in evidence_rows), default=0.0)
        strat = _pick_strategy(m, value_strat)
        decision, prob, traces = strat.decide(
            market=market_dict, evidence=evidence,
            market_yes_price=float(yes_price),
            book_depth_yes_usd=depth_yes, book_depth_no_usd=depth_no,
            bankroll_usd=bankroll, daily_used_usd=daily_used,
            signal_volatility=signal_z,
        )
        save_oracle_calls(traces, m.id)
        save_decision(decision, bankroll_usd=bankroll, daily_used_usd=daily_used, is_live=is_live)

        sz = decision.sizing
        if print_table:
            table.add_row(
                (m.question or "")[:60],
                f"{decision.committed_p:.3f}",
                f"{yes_price:.3f}",
                f"{sz.edge_bps/100:+.1f}",
                sz.side,
                f"{float(sz.notional_usd):.2f}",
                str(len(evidence)),
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
            res = await asyncio.to_thread(executor.place, req)
            save_order(
                decision_id=decision.decision_group_id, market_id=m.id, token_id=token,
                side=sz.side, price=sz.price, size_shares=shares,
                notional=sz.notional_usd, result=res,
            )
            if res.ok:
                daily_used = increment_daily_used(sz.notional_usd)

    if print_table:
        console.print(table)
        console.print(f"[dim]daily notional used: ${float(daily_used):.2f}[/dim]")
    # Push all writes from this pass back to Turso.
    sync_database_replica()
    return by_market


async def _scan_once(*, limit: int, india_only: bool, is_live: bool, max_notional: Decimal,
                     use_news: bool = True) -> None:
    settings = get_settings()
    init_db()
    clob = ClobWrapper(settings)
    oracle = ClaudeOracle(settings)
    value = ValueStrategy(oracle, settings)
    executor = Executor(clob, settings)
    bankroll = Decimal(str(max_notional))
    await _scan_pass(
        settings=settings, clob=clob, oracle=oracle, value_strat=value, executor=executor,
        limit=limit, india_only=india_only, is_live=is_live, bankroll=bankroll, use_news=use_news,
    )


# --- daemon -------------------------------------------------------------


async def _daemon(
    *,
    limit: int, india_only: bool, is_live: bool, max_notional: Decimal,
    rescan_seconds: int, snapshot_seconds: int, hotpath_seconds: int,
) -> None:
    settings = get_settings()
    init_db()
    clob = ClobWrapper(settings)
    oracle = ClaudeOracle(settings)
    value = ValueStrategy(oracle, settings)
    news_event = NewsEventStrategy(inner=value)
    executor = Executor(clob, settings)
    bankroll = Decimal(str(max_notional))

    stop = asyncio.Event()

    def _signal_handler(_sig, _frame=None):  # type: ignore[no-untyped-def]
        log.info("daemon.signal_received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - windows
            signal.signal(sig, _signal_handler)

    last_evidence: dict[str, list[EvidenceRow]] = {}

    async def rescan_loop() -> None:
        while not stop.is_set():
            log.info("daemon.rescan.start")
            try:
                last_evidence.update(
                    await _scan_pass(
                        settings=settings, clob=clob, oracle=oracle, value_strat=value,
                        executor=executor, limit=limit, india_only=india_only,
                        is_live=is_live, bankroll=bankroll, use_news=True, print_table=False,
                    )
                )
            except Exception as e:  # pragma: no cover
                log.error("daemon.rescan.failed", err=str(e))
            await _wait(stop, rescan_seconds)

    async def hotpath_loop() -> None:
        """Re-evaluate any market whose latest signal exceeded news_event z threshold."""
        while not stop.is_set():
            await _wait(stop, hotpath_seconds)
            if stop.is_set():
                break
            hots = hot_markets(last_evidence, z_threshold=news_event.z_threshold)
            if not hots:
                continue
            log.info("daemon.hotpath.fire", markets=hots[:5], count=len(hots))
            try:
                await _scan_pass(
                    settings=settings, clob=clob, oracle=oracle, value_strat=value,
                    executor=executor, limit=limit, india_only=india_only,
                    is_live=is_live, bankroll=bankroll, use_news=True, print_table=False,
                    only_market_ids=set(hots),
                )
            except Exception as e:  # pragma: no cover
                log.error("daemon.hotpath.failed", err=str(e))

    async def snapshot_loop() -> None:
        """Hourly: snapshot every active market for backtest fuel."""
        while not stop.is_set():
            await _wait(stop, snapshot_seconds)
            if stop.is_set():
                break
            log.info("daemon.snapshot.start")
            try:
                async with GammaClient(settings) as gamma:
                    markets = await gamma.list_markets(limit=limit)
                for m in markets:
                    if not m.yes_token_id or m.closed:
                        continue
                    snap = snapshot_market(clob, m.id, m.yes_token_id)
                    save_snapshot(snap)
            except Exception as e:  # pragma: no cover
                log.error("daemon.snapshot.failed", err=str(e))

    async def reconcile_loop() -> None:
        while not stop.is_set():
            await _wait(stop, snapshot_seconds * 4)  # less frequent than snapshots
            if stop.is_set():
                break
            try:
                await pull_resolved_outcomes()
                await asyncio.to_thread(reconcile_fills, clob)
                await asyncio.to_thread(fill_calibration_points)
            except Exception as e:  # pragma: no cover
                log.error("daemon.reconcile.failed", err=str(e))

    log.info(
        "daemon.start", rescan=rescan_seconds, snapshot=snapshot_seconds,
        hotpath=hotpath_seconds, live=is_live, india_only=india_only,
    )
    await asyncio.gather(rescan_loop(), hotpath_loop(), snapshot_loop(), reconcile_loop())


async def _wait(stop: asyncio.Event, seconds: int) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


# --- click CLI ----------------------------------------------------------


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
@click.option("--no-news", is_flag=True, default=False, help="skip RSS+NLP pre-pass")
def cmd_scan(limit: int, india_only: bool, max_notional: str | None, no_news: bool) -> None:
    """One-shot dry-run scan."""
    s = get_settings()
    cap = Decimal(max_notional) if max_notional else s.max_notional_per_day
    asyncio.run(_scan_once(limit=limit, india_only=india_only or s.india_only,
                           is_live=False, max_notional=cap, use_news=not no_news))


@cli.command("trade")
@click.option("--limit", default=50)
@click.option("--india-only", is_flag=True, default=False)
@click.option("--max-notional", required=True)
@click.option("--live/--no-live", default=False)
@click.option("--yes-i-am-sure", is_flag=True, default=False, help="bypass confirmation prompt")
def cmd_trade(limit: int, india_only: bool, max_notional: str, live: bool, yes_i_am_sure: bool) -> None:
    """One-shot live trading."""
    s = get_settings()
    cap = Decimal(max_notional)
    if live:
        if cap > Decimal("500"):
            raise click.UsageError("--max-notional > $500 requires editing the source as a safety check.")
        pre = preflight(s)
        if not pre.ok:
            console.print(pre.explain())
            raise click.UsageError("Allowance preflight failed; refusing to trade.")
        if not yes_i_am_sure:
            confirm = click.prompt(f"Type the cap exactly to confirm live trading (max ${cap})", type=str)
            if confirm.strip() != str(cap):
                raise click.UsageError("Confirmation did not match.")
    asyncio.run(_scan_once(limit=limit, india_only=india_only or s.india_only,
                           is_live=live, max_notional=cap, use_news=True))


@cli.command("run")
@click.option("--limit", default=50)
@click.option("--india-only", is_flag=True, default=False)
@click.option("--max-notional", required=True)
@click.option("--live/--no-live", default=False)
@click.option("--rescan-seconds", default=600, help="seconds between full rescans (default 10 min)")
@click.option("--snapshot-seconds", default=3600, help="seconds between snapshot sweeps (default 1 h)")
@click.option("--hotpath-seconds", default=60, help="seconds between hot-path checks (default 1 min)")
@click.option("--yes-i-am-sure", is_flag=True, default=False)
def cmd_run(limit: int, india_only: bool, max_notional: str, live: bool,
            rescan_seconds: int, snapshot_seconds: int, hotpath_seconds: int,
            yes_i_am_sure: bool) -> None:
    """Long-running daemon (rescan + hot-path + snapshots + reconcile)."""
    s = get_settings()
    cap = Decimal(max_notional)
    if live:
        pre = preflight(s)
        if not pre.ok:
            console.print(pre.explain())
            raise click.UsageError("Allowance preflight failed; refusing to trade.")
        if not yes_i_am_sure:
            confirm = click.prompt(f"Type the cap exactly to confirm LIVE daemon (max ${cap})", type=str)
            if confirm.strip() != str(cap):
                raise click.UsageError("Confirmation did not match.")
    asyncio.run(
        _daemon(
            limit=limit, india_only=india_only or s.india_only, is_live=live, max_notional=cap,
            rescan_seconds=rescan_seconds, snapshot_seconds=snapshot_seconds,
            hotpath_seconds=hotpath_seconds,
        )
    )


@cli.command("reconcile")
def cmd_reconcile() -> None:
    """Pull resolved outcomes from Gamma; reconcile fills; fill CalibrationPoints."""
    init_db()
    clob = ClobWrapper(get_settings())
    asyncio.run(pull_resolved_outcomes())
    inserted = reconcile_fills(clob)
    created = fill_calibration_points()
    summary = daily_brier_summary()
    sync_database_replica()
    console.print(f"[green]fills inserted={inserted} calibration points created={created}[/green]")
    console.print(summary)


@cli.command("backtest")
@click.option("--from", "from_", required=True)
@click.option("--to", "to", required=True)
@click.option("--strategy", default="value")
@click.option("--capital", default="1000")
def cmd_backtest(from_: str, to: str, strategy: str, capital: str) -> None:
    """Walk-forward replay of historical snapshots through the live pipeline."""
    from polyclaude.backtest.harness import run as run_backtest

    start = datetime.fromisoformat(from_).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(to).replace(tzinfo=timezone.utc)
    metrics = run_backtest(start=start, end=end, strategy=strategy, capital=Decimal(capital))
    console.print(metrics)


@cli.command("shadow")
@click.option("--days", default=7)
@click.option("--baseline", default="0.20", help="max allowed Brier; exit non-zero if exceeded")
def cmd_shadow(days: int, baseline: str) -> None:
    """Re-score the last N days of resolved markets and check Brier vs baseline."""
    from polyclaude.backtest.shadow import run as run_shadow

    out = run_shadow(days=days)
    console.print(out)
    if out.get("brier", 0) > float(baseline):
        raise SystemExit(2)


if __name__ == "__main__":
    cli()
