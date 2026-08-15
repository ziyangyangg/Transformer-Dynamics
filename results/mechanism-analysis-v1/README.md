# Mechanism analysis v1

This directory is derived entirely from the immutable snapshot tables in:

- `results/primary-adamw-mechanisms-v1`
- `results/replication-sgd-mechanisms-v1`

Reproduce every file and the Markdown report from the repository root:

```bash
PYTHONPATH=src python -m routing_lab.mechanism_analysis
```

The production defaults use 20,000 paired-seed bootstrap resamples with RNG seed
`20260815`. The independent unit is a training seed. Evaluation episodes, layers,
heads, and checkpoints are never counted as independent observations.

## Tables

- `seed_step_metrics.csv`: one row per optimizer/cell/training-seed/checkpoint;
- `site_step_metrics.csv`: the corresponding long table at each module/layer/head;
- `cell_step_summary.csv`: descriptive seed means and standard deviations over time;
- `paired_delta_summary.csv`: paired final-minus-initial changes, both
  intention-to-train and gate-qualified;
- `site_delta_summary.csv`: the same paired changes for each layer/head site;
- `optimizer_replication.csv`: AdamW/SGD direction agreement on common eligible seeds;
- `functional_gates.csv` and `.json`: registered function/donor gates plus an explicitly
  exploratory target-edge + attention screen. Registered causal `S_key` was not evaluated
  because distractor edges were not blocked one by one;
- `analysis_manifest.json`: sources, row-count audit, bootstrap contract, and numerical
  checks;
- `mechanism_summary.json`: compact machine-readable gates and optimizer comparisons.

`claim_eligible` means function-qualified for routing/attention/Walsh quantities and
function-plus-donor-qualified for QK/OV/FFN quantities. A positive local diagnostic is
not labeled causal compensation: the v1 inputs do not contain finite-output validation,
the FFN practical-floor statistic, or the registered isotropic OV attenuation estimand.
