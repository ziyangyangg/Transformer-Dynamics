# Final report chart map and QA record

This file is the compact design/audit companion for `reports/artifact.json`. The
portable report is the primary reader surface; PNG/SVG figures remain reproducible
high-resolution supplements.

| Report question | Visual | Dataset grain | Intended takeaway | Canonical source |
|---|---|---|---|---|
| Did the registered schedules remove on-manifold cross-talk? | Grouped schedule bar + exact paired table | architecture mean over 10 paired training seeds; 2,048 episodes within seed | Same-LR extension improves cells 3/7 but does not fully clear the gate; lower LR is not a general remedy | `results/scaling-remedy-analysis-b2048-v1/paired_cell_effects.csv` and three mechanism endpoint tables |
| What exploratory architecture patterns appear in compressed dictionary rank? | Zero-centered signed bar + interval table | within-seed factorial contrast, then 10-seed bootstrap | Seven selected secondary contrasts use unadjusted pointwise intervals; the observed load/head signs are discovery targets, not confirmatory effects | `results/scaling-analysis-v1/factorial_effects.csv` |
| How does label-conditioned geometry change through depth? | Ordered line chart | episode→seed→architecture-stratum mean | Query–target selectivity rises through the residual stream | `results/scaling-analysis-v1/representation_geometry_summary.csv` |
| Is that selectivity only global consensus? | Matched ordered line chart | same as above | Global mean cosine stays lower, ruling out only the simplest single-point-consensus story, not multi-cluster structure | same representation table |
| What local geometry surrounds failed and successful checkpoints? | Matched 25×25 heatmaps | one common-initialization diagnostic seed and fixed probe | Plateau and tuned trajectories occupy different local slices; this is a mechanism case, not a population LR effect | `results/dynamics-analysis-v1/loss_landscape_cells.csv` |
| Does the Perspective clustering baseline imply selective routing? | Two-statistic trajectory | 64 fixed-parameter particles over 151 time points | Tokens reach consensus while normalized attention entropy tends to one; clustering is not task-selective routing | `results/clustering-baseline-v1/trajectory.csv` |

## Visual QA

- Titles state the estimand rather than a conclusion; the adjacent prose owns the interpretation.
- Error bars/intervals use training seeds as the independent unit. Episodes, heads,
  layers, cells, and checkpoints are never counted as extra independent samples.
- The remedy plot uses a logarithmic error scale and shows all paired seed paths,
  the mean, and the registered threshold.
- Representation cosines are explicitly labeled descriptive, not causal.
- Rank contrasts are explicitly labeled selected, secondary, exploratory, and
  unadjusted; no visual language implies BH/family-adjusted confirmation.
- Loss landscapes share a logarithmic color normalization and show the checkpoint
  center; the report states the one-seed and two-direction-slice limitations.
- The clustering figure displays both spectral collapse and attention entropy so a
  reader cannot mistake consensus for selective routing.
- Four publication figures were rendered and inspected for clipping, overlap,
  legend ambiguity, and claim/scale consistency. No blocking visual defect remained.

## Portable-reader verification

The canonical artifact passed schema/source validation and packaging. The current
machine did not provide the compatible `chrome-headless-shell` expected by the
dependency-free responsive verifier. Installed full Chrome was attempted, but its
DOM dump exceeded the verifier's bounded output limit. Therefore the receipt records
`structural_only`; this is not represented as a successful browser-interaction test.
