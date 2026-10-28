# Changelog

## 0.2.0 (unreleased)

- Add `evaluate_detailed` and `evaluate_detailed_async`: per-input success,
  uncertainty, Pass@k, cost/latency evidence and failed outputs with error details.
- Add `compare_agents`: named baselines, materialized shared inputs, paired
  bootstrap success-rate intervals and per-input improvement/regression counts.
- Add standalone `pass_at_k`, `wilson_interval`, and `bootstrap_interval` methods.
- Add `entropy run --detailed --pass-k` and `entropy benchmark --paired`.
- Add `evaluate_async` and `Suite.run_async` with sequential trials and cancellation.
- Improve package discovery metadata, documentation links, canonical URLs and sitemap.

### Behavior changes

- `confidence_interval` now reports a pooled 95% Wilson score interval rather
  than an average of per-input Wald intervals. All-success and all-failure data
  retain uncertainty; no observations return `[0.0, 1.0]`. Existing stored baselines
  and thresholds may need review. Pooled bounds assume independent trials.
- Reject nonpositive, boolean and noninteger trial counts before agent calls.
- Empty-message agent exceptions count as failures and skip output checkers.
- Synchronous evaluation rejects awaitables with a message directing callers
  to the async API, rather than silently treating a coroutine as an answer.

Detailed reports can contain raw inputs and failed outputs. Inspect them before
exporting or sharing. Paired bootstrap intervals are exploratory and do not
correct for multiple comparisons or unobserved variation.
