# Training-Aware Transformer Dynamics

## Research question

Can we derive, from a task distribution and gradient flow, the \(QK/OV\) interaction
kernel learned by a full softmax Transformer, then prove that its layer dynamics
implement the interaction graph required by the task?

\[
(\mathcal D,R,\theta_0)
\xrightarrow{\text{training time }s}
\{B_{\ell h}(s),C_{\ell h}(s)\}
\xrightarrow{\text{softmax}}
\mathcal K_{\ell,s}
\xrightarrow{\text{depth }\ell}
\Phi_{\theta_s}^{L}(X),
\]

\[
B_{\ell h}=Q_{\ell h}^{\top}K_{\ell h},
\qquad
C_{\ell h}=O_{\ell h}V_{\ell h},
\qquad
\mathcal K_{\ell,s}(i,j;X)
=\sum_h a_{\ell h,ij}(s;X)C_{\ell h}(s).
\]

Training time and network depth are different variables. The project is complete only
when one theorem connects both arrows. Attention heat maps, low rank, clustering,
superposition, and module patches are not substitute research questions.

The authoritative scope is:

- [Research charter: variables, theorems, data, completion](reports/RESEARCH_CHARTER.md)
- [Source-verified literature and theory map](reports/LITERATURE_MAP.md)
- [Method specification](SPEC.md)
- [Experiment positioning](reports/EXPERIMENT_POSITIONING.md)
- [Implementation plan](tasks/plan.md)
- [Machine-checked repository scope](REPOSITORY_SCOPE.toml)

## What is established

The exact-softmax toy model provides a task with a known correct interaction graph and
fully observable \(E,Q,K,V,O\) training trajectories. A conditional bridge theorem
shows that low risk forces positive slot-blocking selectivity under value-blind scores,
nonnegative gains, and no bypass; a signed-gain construction gives an exact counterexample
when those assumptions are removed.

The multi-seed Phase-II experiment rules out a stable irreducible residual: longer or
scheduled training continues to improve. Dense composites reduce leakage strongly,
whereas rank-matched direct coordinates do not; this is evidence for a
rank/function-class boundary, not a theorem about low-rank attention.

Pythia-70M float64-v4 passes all 8 checkpoint audits. It validates tokenization,
edge-blocking, head observation, and finite-patch instrumentation. It is one pretraining
trajectory, so it does not establish a population training law. Its routing effects are
nonmonotone and template-dependent, and it does not support the proposed
diffuse-to-selective-to-sparse-collision story.

## Active repository

The active package contains four layers:

| Layer | Modules | Purpose |
|---|---|---|
| Task/model | data, model, controlled_model, model_variants | exact data law and full softmax \(QK/OV\) model |
| Training | controlled_training, phase2_study | deterministic gradient-based trajectories and checkpoints |
| Theory measurements | metrics, interventions, phase2_analysis, population_gf | risk, kernel alignment, blocking, exact population updates |
| External validation | pretrained_bridge, pretrained_causal, pretrained_study, pretrained_analysis | audited Pythia checkpoint measurements |

Only two experiment configurations remain active:

- configs/phase2/discovery-remedy/residual-factorization-noffn.json
- configs/pretrained_pythia70m_suite_a_calibration_float64_v4.json

Only six evidence directories remain under results. They are the fixed-kernel baseline,
the Phase-II source/precision/analysis chain, and the Pythia-v4 source/analysis chain.
The exact allowlist is enforced by tests/test_repository_scope.py.

Historical scaling, NTK/landscape, localization, mechanism-attribution, and report-builder
code was removed from the active tree. It remains recoverable at Git commit 1f06157.

## Install and verify

Python 3.11 is required.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[pretrained]"
    PYTHONPATH=src python -m unittest discover -s tests -v

The CI additionally builds and imports a non-editable wheel.

Useful read-only or reproducible entry points:

    PYTHONPATH=src python -m routing_lab.clustering_baseline --help
    PYTHONPATH=src python -m routing_lab.phase2_precision_audit --help
    PYTHONPATH=src python -m routing_lab.phase2_results --help
    PYTHONPATH=src python -m routing_lab.pretrained_analysis --help

Pythia calibration is cache-only unless network access is explicitly requested:

    PYTHONPATH=src python -m routing_lab.pretrained_study \
      --config configs/pretrained_pythia70m_suite_a_calibration_float64_v4.json \
      --output-directory results/pretrained-pythia70m-suite-a-calibration-float64-v4 \
      --cache-directory /path/to/huggingface/cache

## Evidence rules

- The independent inferential unit is a training seed.
- Checkpoints, templates, layers, heads, and prompts are repeated measurements.
- Accuracy alone does not identify an internal interaction kernel.
- A causal statement requires an explicit intervention and descendant recomputation.
- Local hybrid patches are not additive module attribution.
- Failed numerical gates remain failures; thresholds are not relaxed after inspection.
- New experiments must measure \(B_s,C_s,\gamma_s\), transport error, or depth error.

The repository currently grants no third-party reuse license. Public visibility does not
by itself grant permission to copy, modify, run, or redistribute the work.
