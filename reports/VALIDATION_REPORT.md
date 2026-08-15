# Publication validation report

Date: 2026-08-15  
Scope: curated public inspection and owner-reproduction snapshot

## Outcome

The frozen code, mathematical reports, aggregate observations, figures, and the minimal
checkpoint evidence required by the four published dynamics cases passed the publication
gates below. The repository intentionally publishes only 15 registered dynamics snapshots;
all other local model checkpoints and raw record directories remain excluded.

## Executed checks

- Full Python test suite: **137/137 passed** under Python 3.11.
- Publication-focused report, dynamics, scaling, and remedy tests: **38/38 passed**.
- Ruff checks and format checks for every source and test file changed during publication
  review: passed.
- Python bytecode compilation for `src/` and `tests/`: passed.
- Installed dependency consistency (`pip check`): no broken requirements.
- Portable report JSON validation and deterministic serialization: passed.
- Portable HTML packaging: validation and packaging passed. Browser verification is
  structural-only because a compatible headless Chromium binary was unavailable; the four
  central static figures received explicit visual QA.
- Secret and absolute-local-path scan over public code, reports, tests, tasks, and aggregate
  results: no match.
- Public-bundle reconstruction test: the aggregate trajectory tables rebuild the scaling
  analysis without private checkpoint directories.
- Dynamics evidence chain: all 15 retained snapshots match the registered path, cell, seed,
  step, and SHA-256 values in the four source manifests; derived arrays and report artifacts
  are hash checked.
- Scaling, remedy, and dynamics derived-artifact manifests: current file sizes and SHA-256
  values match their recorded provenance.

The test suite emits one known PyTorch warning in a test assertion that converts a
gradient-bearing tensor to a scalar. It does not change the computed value or any test
outcome.

## Claim audit

The publication language has been checked against the actual estimands:

- The exact Walsh--Parseval result is a function-level causal-forcing identity. It does not
  uniquely identify QK, OV, FFN, a head, or a factorization.
- The measured QK midpoint decomposition is an exploratory protocol deviation from the
  preregistered asymmetric content/route/interaction split. It does not test or refute the
  preregistered QK claim.
- The evaluator blocked the target query edge but did not separately block every distractor
  edge. The registered causal key selectivity `S_key` is therefore unevaluated; the published
  target-edge-plus-attention quantity is only an exploratory screen.
- The b=2,048 remedy follow-up reuses selected cells and seeds and reports unadjusted
  pointwise intervals. It is targeted exploratory evidence, not independent confirmatory
  inference.
- The seven normalized-rank contrasts are secondary, selected, and reported with
  unadjusted pointwise intervals; no BH family correction was applied. They are exploratory
  architecture patterns.
- The so-called width contrast changes both d and C at fixed C/d; it is not an isolated
  width effect.
- The common-initialization loss-landscape/NTK comparison is a one-seed mechanism case
  study, not a population causal estimate of learning rate.
- The clustering reproduction uses fixed parameters and demonstrates that global consensus
  can coexist with uniform attention; it is not a training-dynamics result.
- Confirmed downstream compensators remain **zero**. OV selectivity and FFN cancellation are
  candidates requiring finite, replicated module-local causal gates.

## Reuse boundary

`LICENSE` intentionally grants no third-party permission to copy, modify, distribute, or
run the work. The public repository is therefore an inspection and owner-reproduction
handoff, not an open-source release. A permissive or research license must be chosen by the
owner before describing the repository as third-party reproducible software.
