# Phase-II results report

- Cohort: `discovery-remedy` (never pooled across cohorts)
- Independent inference unit: **training seed**
- Whole-seed bootstrap resamples: 20,000
- Validated source studies: 1

## What was validated

Every root config hash, root/seed manifest, checkpoint-state schedule, and `(study_config_hash, cell_hash, seed, step)` primary key was checked before analysis. Root checkpoint rows were required to equal the union of committed seed rows exactly.

## Training-limit question

- `phase2-residual-factorization-noffn-discovery-remedy-v2` / `hard-factorized-constant-6400`: late-rate TOST `False`, plateau gate `False`, practical-floor gate `False`.
- `phase2-residual-factorization-noffn-discovery-remedy-v2` / `hard-factorized-cosine-6400`: late-rate TOST `False`, plateau gate `False`, practical-floor gate `False`.

## Factorization interpretation guardrail

Rank-matched direct is the same-function-class conditioning control. Dense direct is a rank/function-capacity upper bound. Dense-only improvement must not be called pure optimization geometry. The complete accuracy/risk/Xi_value gate is reported per arm and per training seed; passing it does not turn a capacity upper bound into a conditioning result.

Registered P19 classification: `rank_or_function_capacity`; inference boundary: `discovery_only_not_confirmation`.

## Exploratory matrices

Representation 2×2 effects and head-family slopes are exploratory. Their paired-seed sign-flip p-values are BH-adjusted at q=0.10 across the exact family reported in `analysis_summary.json`; plotted confidence intervals are unadjusted pointwise seed-bootstrap intervals.

## Figure reading

Thin lines or dots are individual training seeds. Opaque estimates and bands are seed means and labeled confidence intervals; trajectory bands are explicitly pointwise 95% visualization intervals. Checkpoints, heads, and episodes never increase the reported sample size.
