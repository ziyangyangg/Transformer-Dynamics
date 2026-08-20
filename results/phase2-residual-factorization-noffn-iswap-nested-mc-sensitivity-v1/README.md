# Phase-II I_swap nested-MC sensitivity audit

## Result first

This is a **sensitivity/discovery audit**, not a replacement for preregistered P8 and
not a new confirmation cohort.  It re-evaluates 144 logical arm-step
states (132 unique checkpoint byte hashes) on 256 independent blocks x
2,048 IID support-preserving swaps =
524,288 pairs per checkpoint.

* Final all-state precision gate: **FAIL**.
* Dense-direct hard-arm P19 result survives both outer and hierarchical inference:
  **True**.
* Rank-matched non-remedy survives both: **True**.
* Cosine-minus-constant I_swap slope direction survives: **True**.
* Outer classification: `rank_or_function_capacity`; hierarchical classification:
  `rank_or_function_capacity`.

## What was estimated

For checkpoint theta and IID episode X, let X_swap replace one non-target concept
with a uniformly sampled absent concept while leaving its value and the target
unchanged:

`D_theta(X) = (f_theta(X_swap)-f_theta(X))^2`,
`I_swap(theta) = E[D_theta(X)]`.

The same serialized `(training seed r, block k)` episode population is used at every
arm and step.  Therefore block-paired differences measure conditional Monte Carlo
error without adding an intervention-stream confound.  The stream key is exactly
`SHA256([study_hash, "iswap-mc-v1", r, k])[:8]` interpreted as a 63-bit integer.

## Precision and tails

* High-N checkpoint RSE median/max: 0.0206 / 0.2927; gate <= 0.10.
* Block-extrapolated b=2,048 RSE median: 0.330.  This
  is an extrapolation from independent block dispersion, not a retroactive SE for
  the single registered draw.
* Maximum paired block-bootstrap SE across log2 contrasts/slopes: 0.4068 bit; gate <= 0.25 bit.
* Median Gini / median effective-sample fraction: 0.976 / 0.005.
* Median [10%,90%] log2(high-N / registered): 0.036
  [-0.480, 1.077] bit.

Every checkpoint also stores CV(D), n_eff=(sum D)^2/sum(D^2), top-1/top-10 episode
shares, top-1%/top-10% shares, Gini, and cumulative K=8/16/32/64 (plus extension
stages if reached).  `tail_triads.csv` is an explicitly exploratory, nonblocking
table keyed by ordered `(query q, old distractor c, absent donor c')`, distractor
slot/value, target slot, and label.  It exists to support later regressions against
learned E-Gram/QK/OV geometry; it does not replace the IID primary analysis.

## Inference

`nested_inference.json` contains two 20,000-draw analyses:

1. outer-only whole-training-seed resampling;
2. hierarchical seed + block resampling, with one block-index vector reused across
   all arms/steps in a selected seed occurrence.

Inside every draw the code recomputes log2 factorization contrasts, four-point
800/1600/3200/6400 slopes, q=log2(I_6400/I_3200), practical-floor indicators, and
the nine-column P19 family (three comparisons x R/L_W/I_swap).  R and L_W are exact
frozen source measurements; only I_swap receives inner MC resampling.  P19 also
retains risk/accuracy/Xi noninferiority and the >=80% per-seed function gate.

## Reproduction and boundaries

Run `python -m routing_lab.phase2_swap_sensitivity --source-directory ...
--output-directory ... --device cuda`.  Per-seed NPZ files contain numeric episode
metadata and raw float64 D, with pickle disabled.  `_SUCCESS` is written last;
source files, code files, logical rows, checkpoint bytes, NPZ files, tables, plots,
and reports are SHA-256 bound in `artifact_manifest.json`.

The study uses already-observed discovery-remedy seeds 100..111.  Passing this audit
can show that a prior direction is not an artifact of b=2,048 MC noise.  It cannot
promote the result to confirmation, establish total causal mediation, or interpret
dense-only improvement as pure Q/K/O/V factorization-conditioning evidence.
