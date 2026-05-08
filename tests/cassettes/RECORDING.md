# Recording cassettes against live APIs

The `gamma_list_markets.yaml` cassette in this directory is hand-crafted to
exercise every branch of `polyclaude/markets/gamma.py:_parse_market` and
`polyclaude/ledger/reconcile.py:_extract_outcome`. It mirrors Gamma's real
response schema but adds synthetic edge cases (closed markets in three
different forms, JSON-string-encoded outcome arrays, India-context tags).

## Why hand-crafted not recorded

The synthetic fixture is intentional. Recording against live Gamma has two
problems:

1. Live Gamma response shapes change subtly over time (field names, ordering,
   whether outcomes is an array or a JSON-string). A hand-written fixture that
   exercises every shape we know about gives us defense in depth.
2. Real cassettes change every recording (new top markets, different volumes)
   and produce noisy diffs. Hand-crafted is stable.

If you want to add a real recording alongside the synthetic one, run from any
environment with network access:

```bash
POLYCLAUDE_VCR_RECORD=1 pytest tests/test_gamma_integration.py -v
```

That will save a new cassette to `tests/cassettes/gamma_list_markets.yaml`
overwriting this file. Commit it if it captures behaviour the synthetic
fixture missed.

## Updating tests when the parser grows new branches

When you add a new branch to `_parse_market` or `_extract_outcome`, add a new
synthetic market entry above mirroring the response shape and assert against
it in `tests/test_gamma_integration.py`.
