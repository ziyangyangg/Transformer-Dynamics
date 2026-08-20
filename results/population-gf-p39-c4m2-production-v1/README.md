# C4/m2 population-GF production status

## Outcome

P39 is blocked, not failed. The final prospectively frozen numerical-remedy
triplet \((\eta/16,\eta/32,\eta/64)\) still fails the P38 threshold 0.10 for
validation seeds 201 and 202. Consequently seeds 300--303 were neither generated
nor inspected, no low-dimensional vector field was fit, and no closure score or
nearest-neighbor counterexample was computed.

Seed 201 retains embedding-effective-rank discrepancies 0.146643 and 0.121250.
Seed 202 retains higher-order-leakage discrepancies 0.263352 and 0.175821, plus
Q/K condition-imbalance discrepancies 0.311235 and 0.559178. The corresponding
denominator norms range from 0.0481 to 0.2945, so these failures are not created
by the \(10^{-12}\) denominator floor.

Risk is numerically resolved, but the registered functional-decomposition and
representation coordinates are not. On the final nested pair, risk discrepancies
are only 0.0031 for seed 201 and 0.0133 for seed 202. Seed 201 still fails the
embedding-effective-rank coordinate; seed 202 fails both higher-order functional
leakage and Q/K Gram imbalance. The higher-order-leakage jump for seed 202 moves
between physical times 13.5, 14.1, and 14.7 as the Euler step changes, and the
largest Q/K-imbalance change moves with it. Thus a well-converged scalar loss can
coexist with unresolved functional and representation paths. That is exactly why
P38 is an intersection--union gate over every registered order parameter.

## What the optimizer bridge says

At matched seed 100 and learning rate 0.003, stochastic SGD at step 1600 is very
close to the Euler population-GF checkpoint at physical time 4.8; for example,
risk differs by about \(7.2\times10^{-6}\). AdamW follows a substantially
different adaptive path and reaches risk 0.1712 with nonzero distractor and
higher-order leakage. This is an optimizer-geometry calibration only. AdamW and
SGD are discrete stochastic dynamics, so their differences are neither P38 tests
nor repairs for the failed Euclidean-GF reference.

## Files

- p38_summary.csv: every original and refinement triplet, failed coordinates,
  worst discrepancy, and runtime.
- final_failure_decomposition.csv: numerator and denominator norms for every
  final failed coordinate.
- optimizer_endpoint.csv: the paired optimizer endpoints and matched GF
  checkpoint.
- manifest.json: machine-readable stopping decision and claim boundary.
- ../../reports/P39_SOURCE_INTEGRITY_CONTRACT.md: the recomputation and provenance
  contract applied before any future held-out closure analysis.

The exact registered experiment configuration is
configs/population_gf_p39_c4m2_v1.json. Original failed artifacts were preserved;
no directory was overwritten and the P38 threshold was not relaxed.

This directory is a **convenience status narrative**, not a self-validating
evidence bundle: its summary CSVs and prose were manually assembled after the
strict source replays. The auditable evidence remains the original/refined GF
directories plus the validation contract above. A future publication pass should
replace this convenience bundle with a deterministic builder that records source
hashes, generator identity, and byte receipts for every derived file.

## Next registered numerical work

The next attempt should use an established higher-order ODE remedy rather than
continue halving explicit Euler indefinitely:

1. prospectively freeze fixed-step RK4 and adaptive Dormand--Prince tolerances,
   the same physical observation grid, and the same P37 gate;
2. compare function-space coordinates and representation coordinates separately,
   including signed Gram entries and squared norms in addition to Frobenius norms;
3. localize seed 202's event near physical time 13--15 with a finer observation
   grid, without aligning trajectories post hoc to make the event look converged;
4. only after all validation seeds pass, rerun discovery and fresh untouched
   seeds on one common finest-resolution scheme and then evaluate P39.

Until those standard numerical explanations are exhausted, the residual is not
promoted to a new Transformer-capacity or training-dynamics open problem.
