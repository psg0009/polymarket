# polyclaude

A Polymarket trading agent that uses Claude as a calibrated probability oracle, with multi-modal news + audio ingestion (FinBERT/FinVADER/Whisper), an Indian-markets vertical, and a backtester that reuses the live pipeline.

> Status: full pipeline wired end-to-end (P0+P1+P2 from the spec). Heavy NLP / audio / Streamlit deps are optional and lazy-loaded; minimal installs work.

## Why this design

Three rules drive the architecture and all three are easy to skip and expensive to skip:

1. **No anchoring on price.** The probability call to Claude *never* receives the current market price. Sizing is a separate, second call. (`tests/test_anchoring.py`)
2. **Allowance preflight.** Polygon EOA traders need three exchange-contract approvals before any order can settle. We refuse to trade if those are missing and print the exact `approve` calls. (`polyclaude preflight`)
3. **Backtester reuses live modules.** No separate "research" code path — `backtest/replay.py` walks `BookSnapshot` rows through the same `ValueStrategy.decide()` the live agent uses.

## Layout

```
polyclaude/
├── config.py              # Pydantic settings from .env + Streamlit secrets bridge
├── logging_setup.py
├── clob/                  # client wrapper, allowance preflight, executor with async reprice + idempotency
├── markets/               # Gamma API, snapshot persistence, India tagger
├── ingest/                # RSS / NewsAPI / Whisper audio (audio_scheduler.py polls feeds and transcribes)
├── nlp/                   # FinBERT (lazy), FinVADER, recency-decay ensemble, MiniLM linker (lazy)
├── pipeline/discover.py   # ingest → score → link → persist Events+Signals; feeds evidence into the oracle
├── oracle/                # Three-call pattern: ambiguity → probability (no price) → sizing
├── strategy/              # Fractional Kelly + value / news_event / india_elections
├── ledger/                # Schema, persist helpers, reconcile.py (pulls Gamma resolutions, writes CalibrationPoints)
├── backtest/              # replay (reuses live modules), metrics, harness, shadow.py (CI re-score)
├── ui/dashboard.py        # Streamlit (reads from local SQLite or hosted Turso/Postgres)
└── main.py                # CLI: init / preflight / scan / trade / run / reconcile / backtest / shadow
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                          # core
pip install -e ".[nlp,audio,ui,dev]"      # + FinBERT, Whisper, Streamlit, pytest
cp .env.example .env                      # fill in PRIVATE_KEY, FUNDER, ANTHROPIC_API_KEY
polyclaude init                           # create the SQLite schema
```

### Allowance approval (EOA mode only)

If you sign with a raw EOA (`SIGNATURE_TYPE=1`), you must approve three Polygon exchange contracts to spend USDC and the conditional-token NFTs. From the EOA, on Polygon mainnet:

```
USDC.approve(0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E, MAX_UINT256)
USDC.approve(0xC5d563A36AE78145C45a50134d48A1215220f80a, MAX_UINT256)
USDC.approve(0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296, MAX_UINT256)
CTF.setApprovalForAll(<each of the above>, true)
```

`polyclaude preflight` reads current allowances on-chain and tells you exactly which calls are missing. Proxy users (`SIGNATURE_TYPE=2`/`3`) skip this — approvals live on the proxy.

## CLI

```bash
# One-shot dry-run scan with full RSS+NLP pre-pass (default).
polyclaude scan --limit 50

# India only.
polyclaude scan --india-only

# Long-running daemon — three cadences:
#   * rescan every 10 min (pull markets, run RSS+NLP, score with Claude, place orders)
#   * news-event hot path every 1 min (re-evaluate any market whose latest signal has |z| >= 3)
#   * snapshots every 1 h (persist orderbook depth for backtest fuel)
#   * reconcile every ~4 h (pull resolved outcomes from Gamma, fill CalibrationPoints)
polyclaude run --max-notional 100 --rescan-seconds 600 --hotpath-seconds 60 --snapshot-seconds 3600

# Live trading: the daemon respects --live and confirms the cap before placing real orders.
polyclaude run --live --max-notional 50

# Audio scheduler (separate process; polls audio_sources.txt and transcribes via Whisper).
python -m polyclaude.ingest.audio_scheduler

# Manual reconcile.
polyclaude reconcile

# Backtest.
polyclaude backtest --from 2025-01-01 --to 2025-12-31 --strategy value --capital 1000

# Shadow CI: re-score last 7 days vs. baseline; exit non-zero on regression.
polyclaude shadow --days 7 --baseline 0.20
```

## Calibration philosophy

The oracle runs three calls per market:

1. **Ambiguity.** `clarity < 0.7` → skip. `0.7 ≤ clarity < 0.85` → halve size. `verdict == "hostile"` → skip.
2. **Probability.** Claude **never** sees the current market price (`oracle.evaluate_probability` strips `yes_price` / `midpoint` / `market_p` from the market dict before calling). Evidence rows include explicit `tier=N/OFFICIAL|WIRE|NATIONAL|REGIONAL|SOCIAL` so Claude can weight ECI/PIB > Reuters > regional explicitly.
3. **Sizing.** Sees the price + book depth + signal volatility. Computes the edge.

The transition from dry-run to live is gated on Brier ≤ 0.20 over ≥ 100 resolved markets — a single SQL query against `calibration_points` tells you whether you've earned the right. The Streamlit dashboard's "Live status" tab shows this gate in real time.

## Always-on dashboard (Path A — Streamlit Cloud + Turso)

The agent runs locally / on a tiny VM and writes to a hosted DB; the Streamlit Cloud dashboard reads from the same DB and stays live 24/7.

### Step 1. Provision Turso (~2 min)

```bash
# Install the Turso CLI (https://docs.turso.tech/cli/installation)
brew install tursodatabase/tap/turso          # macOS
# or:  curl -sSfL https://get.tur.so/install.sh | bash

turso auth signup
turso db create polyclaude
turso db show polyclaude --url               # → libsql://polyclaude-<org>.turso.io
turso db tokens create polyclaude            # → eyJhbGc...
```

Compose the URL:
```
sqlite+libsql://polyclaude-<org>.turso.io/?authToken=eyJhbGc...
```

### Step 2. Initialise the schema and run the agent locally pointing at Turso

```bash
export DATABASE_URL='sqlite+libsql://polyclaude-<org>.turso.io/?authToken=...'
polyclaude init
polyclaude run --max-notional 25         # the agent writes Events / OracleCalls / Decisions to Turso
```

### Step 3. Deploy the dashboard to Streamlit Cloud (~3 min)

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Click *New app* → repo `psg0009/polymarket`, branch `claude/polymarket-trading-agent-Wo9yk`, **Main file path** `polyclaude/ui/dashboard.py`.
3. *Advanced settings* → **Secrets** → paste:
   ```toml
   DATABASE_URL = "sqlite+libsql://polyclaude-<org>.turso.io/?authToken=..."
   ```
4. Deploy. Public URL is `https://polyclaude-<random>.streamlit.app`.

The dashboard re-deploys on every push to the chosen branch. If you only need the dashboard reading data, your agent process is the only thing that needs Turso write credentials — the Streamlit Cloud app uses the same URL but only reads.

## Ledger schema (chain of custody)

```
Event ──► Signal ──► OracleCall ──► Decision ──► Order ──► Fill ──► PnLSnapshot
                                       ▲
                                       └── CalibrationPoint  (filled at resolution)
```

Every link foreign-keys to the prior step. A losing trade can be replayed all the way back to the news article (or audio transcript) that triggered it, the exact prompt sent to Claude, and the response that justified the size. `decision_group_id` chains the up-to-three Claude calls (ambiguity → probability → sizing) that produce one Decision.

## Indian markets vertical

Markets resolving on Indian elections, RBI rate decisions, IPL/cricket, IPOs, and Nifty/Sensex are auto-tagged via word-boundary regex (`markets/india.py`). India election markets route through `IndiaElectionStrategy`, which adds an IST-aware exit-poll embargo so the daemon refuses to place new orders during polling hours.

Source weighting (`ingest/normalizer.py`):

| Tier | Examples |
| ---- | -------- |
| 1 (official) | ECI, PIB, RBI, Federal Reserve, ec.europa.eu |
| 2 (wire)     | Reuters, AP, Bloomberg, AFP, PTI |
| 3 (national) | NYT, WSJ, BBC, The Hindu, Indian Express, NDTV, ToI |
| 4 (regional) | News18, The Quint, Scroll |
| 5 (social)   | Twitter/X, Reddit |

Each evidence row passed to Claude includes `tier=N/<LABEL>` so the oracle can weight by hand. The probability prompt explicitly tells Claude this hierarchy.

## Tests + CI

```bash
pytest                            # 50+ unit tests
POLYCLAUDE_E2E=1 pytest -k smoke  # opt-in real-API smoke test (needs ANTHROPIC_API_KEY)
```

Two GitHub Actions workflows ship:

- `.github/workflows/tests.yml` — runs pytest on every push/PR.
- `.github/workflows/shadow.yml` — daily cron + on-PR; re-scores last 7 days of resolved markets and **fails the build if Brier exceeds the baseline** (default 0.20).

The shadow job is what enforces calibration-aware merge gating: if a prompt or strategy change worsens Brier on real history, the PR can't merge.

## Optional dependencies

- `polyclaude[nlp]` — `transformers`, `torch`, `sentence-transformers` for FinBERT and the MiniLM linker.
- `polyclaude[audio]` — `faster-whisper` for the audio scheduler.
- `polyclaude[ui]` — `streamlit`, `pandas`, `matplotlib`.
- `polyclaude[dev]` — `pytest`, `vcrpy`, `ruff`, `mypy`.

The core agent runs without any of these — FinBERT degrades to neutral scores, the linker degrades to keyword Jaccard, and the dashboard is simply unavailable.
