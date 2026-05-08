# polyclaude

A Polymarket trading agent that uses Claude as a calibrated probability oracle, with multi-modal news + audio ingestion (FinBERT/FinVADER/Whisper), an Indian-markets vertical, and a backtester that reuses the live pipeline.

> Status: scaffold complete; dry-run path is end-to-end functional. Heavy NLP / audio / Streamlit deps are optional and lazy-loaded — minimal installs work.

## Why this design

Three rules drive the architecture, and all three are easy to skip and expensive to skip:

1. **No anchoring on price.** The probability call to Claude *never* receives the current market price. Sizing is a separate, second call. This is the single change that most improves calibration vs naive agents (see `tests/test_anchoring.py`).
2. **Allowance preflight.** Polygon EOA traders need three exchange-contract approvals before any order can settle. We refuse to trade if those are missing and print the exact `approve` calls to make. (`polyclaude preflight`)
3. **Backtester reuses live modules.** No separate "research" code path. Any backtest improvement is a live improvement, and every prompt is replayed through the same `ValueStrategy.decide()`.

## Layout

```
polyclaude/
├── config.py              # Pydantic settings from .env
├── clob/                  # py-clob-client wrapper, allowance preflight, executor
├── markets/               # Gamma API, snapshot persistence, India tagger
├── ingest/                # RSS news, NewsAPI, faster-whisper audio, India sources
├── nlp/                   # FinBERT + FinVADER ensemble, MiniLM event↔market linker
├── oracle/                # Claude calls: ambiguity + probability + sizing (3-call pattern)
├── strategy/              # Fractional-Kelly sizing + value / news_event / india_elections
├── ledger/                # SQLAlchemy schema, persist helpers, reconcile job
├── backtest/              # Walk-forward replay reusing live modules + metrics (Brier, calibration, Sharpe)
├── ui/                    # Streamlit dashboard
└── main.py                # `polyclaude` CLI: scan / trade / reconcile / backtest / preflight / init
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # core
pip install -e ".[nlp,audio,ui,dev]"  # optional: FinBERT, Whisper, Streamlit, pytest
cp .env.example .env             # fill in PRIVATE_KEY, FUNDER, ANTHROPIC_API_KEY
polyclaude init                  # create the SQLite schema
```

### Allowance approval (EOA mode only)

If you sign with a raw EOA (`SIGNATURE_TYPE=1`), you must approve three Polygon exchange contracts to spend USDC and the conditional-token NFTs. From the EOA, on Polygon mainnet:

```
USDC.approve(0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E, MAX_UINT256)
USDC.approve(0xC5d563A36AE78145C45a50134d48A1215220f80a, MAX_UINT256)
USDC.approve(0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296, MAX_UINT256)
CTF.setApprovalForAll(<each of the above>, true)
```

`polyclaude preflight` reads the current allowances on-chain and tells you exactly which calls are missing. Proxy users (`SIGNATURE_TYPE=2`/`3`) can skip this step — approvals live on the proxy.

## Usage

```bash
# 1. Dry-run scan: print decision table for the top 50 markets, write the run to SQLite.
polyclaude scan --limit 50

# 2. India only.
polyclaude scan --india-only

# 3. Live trading, capped at $50/day, with a typed-confirmation prompt.
polyclaude trade --live --max-notional 50

# 4. Nightly: reconcile fills, fill CalibrationPoints, print Brier summary.
polyclaude reconcile

# 5. Walk-forward replay against historical snapshots in your ledger.
polyclaude backtest --from 2025-01-01 --to 2025-12-31 --strategy value --capital 1000

# 6. Streamlit dashboard.
streamlit run polyclaude/ui/dashboard.py
```

## Calibration philosophy

The oracle returns three things: a probability, a confidence, and a rationale. Calibration matters more than direction — a well-calibrated agent that says 60% is right ~60% of the time, and that's the goal. Over-confidence at the tails is the most expensive failure mode, so the prompts (`polyclaude/oracle/prompts.py`) explicitly instruct Claude to start from a base rate / reference class and update from there, rather than starting at 0.5 by habit.

We enforce three gates before placing a trade:

1. **Ambiguity gate.** Claude reads the resolution rules first. `clarity < 0.7` skips the market; `0.7 ≤ clarity < 0.85` halves the size; `verdict == "hostile"` always skips.
2. **Edge gate.** `|p_oracle - p_market| ≥ 3¢` (configurable).
3. **Depth gate.** Book depth within 2¢ of midpoint must exceed `MIN_BOOK_DEPTH_USD`.

The transition from dry-run to live is gated on Brier ≤ 0.20 over ≥ 100 resolved markets — a single SQL query against the `calibration_points` table tells you whether you've earned the right.

## Ledger schema (chain of custody)

```
Event ──► Signal ──► OracleCall ──► Decision ──► Order ──► Fill ──► PnLSnapshot
                                       ▲
                                       └── CalibrationPoint  (filled at resolution)
```

Every link foreign-keys to the prior step. A losing trade can be replayed all the way back to the news article (or audio transcript) that triggered it, the exact prompt sent to Claude, and the response that justified the size. `decision_group_id` chains the up-to-three Claude calls (ambiguity → probability → sizing) that produce one Decision.

## Indian markets vertical

`polyclaude/markets/india.py` tags markets resolving on Indian elections, RBI rate decisions, IPL/cricket, IPOs, and Nifty/Sensex. `polyclaude/ingest/india_sources.py` ships the canonical RSS feed bundle (PIB, RBI, PTI, The Hindu, Indian Express, NDTV, Times of India, Mint, Business Standard). The `IndiaElectionStrategy` adds an exit-poll embargo window in IST so the scheduler doesn't trade through polling hours.

Source weighting (in `ingest/normalizer.py`):

| Tier | Examples |
| ---- | -------- |
| 1 (official) | ECI, PIB, RBI, Federal Reserve, ec.europa.eu |
| 2 (wire)     | Reuters, AP, Bloomberg, AFP, PTI |
| 3 (national) | NYT, WSJ, BBC, The Hindu, Indian Express, NDTV, ToI |
| 4 (regional) | News18, The Quint, Scroll |
| 5 (social)   | Twitter/X, Reddit |

The probability prompt explicitly tells Claude this hierarchy.

## Tests

```bash
pytest
```

50 tests covering: Kelly math, cap enforcement, JSON parser fallback, Pydantic schema validation, the no-anchoring guard (oracle.evaluate_probability strips `yes_price` / `midpoint` / `market_p` from the market dict before the API call), the ambiguity gate decision table, idempotency-key behavior, the India tagger, ledger round-trip, ensemble decay, and backtest metrics.

## Optional dependencies

- `polyclaude[nlp]` — `transformers`, `torch`, `sentence-transformers` for FinBERT and the event↔market linker.
- `polyclaude[audio]` — `faster-whisper` for press-conference and earnings-call transcription.
- `polyclaude[ui]` — `streamlit`, `pandas`, `matplotlib` for the dashboard.
- `polyclaude[dev]` — `pytest`, `vcrpy`, `ruff`, `mypy`.

The core agent runs without any of these; FinBERT degrades to neutral scores, the linker degrades to keyword Jaccard, and the dashboard is simply unavailable.

## What's intentionally not here

No backwards-compat shims, no auto-migration, no production deployment recipe. This is a dry-run-first agent intended to be run interactively while you tune the prompts and watch the calibration curve flatten. Once Brier on resolved markets is in the right ballpark, flip `--live` on with a small `--max-notional` cap.
