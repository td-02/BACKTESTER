# nanoback 0.2.0 Release Checklist

## Release Goals

- Tighten the existing engine architecture.
- Keep benchmark performance at or better than the current baseline.
- Add a log book for latency figures so hot paths are visible and comparable.

## Package Metadata

- [ ] Bump the package version to `0.2.0`.
- [ ] Keep the author metadata set to `Tapesh Chandra Das`.
- [ ] Confirm the PyPI project URLs are correct.
- [ ] Verify the release tag matches the package version.

## Architecture Cleanup

- [ ] Remove duplicate validation and configuration logic in hot paths.
- [ ] Keep the data flow for backtest execution explicit and easy to trace.
- [ ] Make timing and tracing hooks optional so normal runs stay lean.
- [ ] Preserve deterministic behavior for repeated benchmark runs.

## Benchmark Targets

- [ ] Run the benchmark script with the standard dataset.
- [ ] Capture elapsed time, fill count, and PnL for each run.
- [ ] Record a baseline for the `0.2.0` release.
- [ ] Compare future runs against the baseline and flag regressions.

## Latency Log Book

- [ ] Record per-stage timings for data loading, signal generation, order creation, risk checks, matching, and ledger writes.
- [ ] Log per-run summaries with `count`, `mean`, `p50`, `p95`, `p99`, and `max`.
- [ ] Emit a machine-readable format such as JSONL or CSV.
- [ ] Include a stable run ID and benchmark seed for replay.
- [ ] Make the log book easy to diff across versions.

## Release Steps

- [ ] Run the test suite.
- [ ] Run the benchmark smoke test.
- [ ] Build the sdist.
- [ ] Confirm the source distribution contains the updated metadata.
- [ ] Push the release tag and verify the GitHub Actions publish run.
- [ ] Check the PyPI project page after publish.

## Feature Ideas for 0.2.x

- [ ] Replay mode for deterministic profiling.
- [ ] Memory and allocation counters.
- [ ] Structured trace export for deeper offline analysis.
- [ ] Benchmark regression gate in CI.
- [ ] Hot-path summaries that identify the slowest stage directly.
