# Training-Aware Transformer Dynamics

## Conditional research question

Under what explicit and checkable conditions does population gradient flow in a
factorized exact-softmax Transformer learn a task-aligned $QK/OV$ interaction kernel,
and when does that learned kernel implement the task-required interaction graph through
network depth?

$$
(\mathcal D,R,\theta_0)
\xrightarrow{\text{training time }s}
\{B_{\ell h}(s),C_{\ell h}(s)\}
\xrightarrow{\text{exact softmax}}
\mathcal K_{\ell,s}
\xrightarrow{\text{network depth}}
\Phi_{\theta_s}^{L}(X),
$$

$$
B_{\ell h}=Q_{\ell h}^{\top}K_{\ell h},
\qquad
C_{\ell h}=O_{\ell h}V_{\ell h},
\qquad
\mathcal K_{\ell,s}(i,j;X)
=\sum_h a_{\ell h,ij}(s;X)C_{\ell h}(s).
$$

This is not a claim for arbitrary data or initialization. The program seeks conditions
that are broad enough to cover more than one task but concrete enough to test:

| Condition | Plain meaning |
|---|---|
| identifiable task graph | changing a required source changes the correct answer |
| finite representability | the available head ranks can express the required messages |
| nondegenerate factor access | factorization does not trap training away from every task-equivalent kernel |

These are the three current candidate gate categories, not final universal assumptions.
Each is extracted from a concrete failure mode in the registered task or model: an
unidentifiable edge, an unattainable task message, or an invariant factorization trap.
Any covariance, margin, or stability constant must be derived from the exact data law
and architecture. MQAR is the one-step proof slice; LEGO tests depth composition.

Training time and network depth are different variables. Attention maps, low rank,
clustering, superposition, and local patches remain evidence or boundary conditions,
not replacement research questions.

Authoritative documents:

- [Research charter](reports/RESEARCH_CHARTER.md)
- [Formal matrix-MQAR proof decomposition](reports/MATRIX_MQAR_PROOF_DECOMPOSITION.md)
- [Main theorem and literature roadmap](reports/MAIN_THEOREM_ROADMAP.md)
- [MQAR kernel-learning theorem](reports/MQAR_KERNEL_LEARNING_THEOREM.md)
- [Minimal matrix MQAR specification](reports/MATRIX_MQAR_C3M2_SPEC.md)
- [Matrix MQAR critical-point result](reports/MATRIX_MQAR_C3M2_RESULT.md)
- [Four-gate MQAR-to-LEGO status](reports/FOUR_GATE_STATUS.md)
- [Literature and theory map](reports/LITERATURE_MAP.md)
- [Method specification](SPEC.md)
- [Experiment positioning](reports/EXPERIMENT_POSITIONING.md)
- [Implementation plan](tasks/plan.md)
- [Machine-checked repository scope](REPOSITORY_SCOPE.toml)

## Established results

On a one-layer, one-head, value-linear MQAR-compatible population, the exact risk
closes in two composite variables: the target score margin $\delta$ and value gain
$g$. For positive nondegenerate factor initialization on the registered symmetric
role-tied parameterization, population gradient flow satisfies

$$
\delta(s)\to+\infty,
\qquad
g(s)\to1,
\qquad
\mathcal E_{\mathcal K}(s)=2R(s)\to0.
$$

The unrestricted claim is false. Zero query/key factors form an exact access barrier.
A signed-gain two-head construction has $R=0$ but $S_{\rm key}=0$. Thus factor access
and route identifiability cannot be omitted.

The $C=3,m=2$ matrix lift is sharper. Correct retrieval forces
$S_{qq}-S_{qd}\to+\infty$, so boundedness of $(S,g)$ and compact LaSalle are
impossible. Balanced full-rank initialization is also insufficient: the invariant
branch $K=-Q$ cannot retrieve every ordered concept pair. Its canonical uniform wrong
boundary is nevertheless a saddle with four analytically certified unstable modes.
The current target is therefore almost-everywhere boundary selection: classify every
wrong access-singular family, prove its attracting set has measure zero, and exclude
unclassified escape in compactified coordinates.
The general matrix task identifies the delivered coefficients $g a_i$, not $g$
alone; zero-risk sequences can have $g\to\infty$.

Only after that theorem closes does the project enter the published LEGO
state-tracking law. The repository now
contains the complete finite cyclic population, a learned local cyclic transition
operator, and an exact finite-depth composition bound. Parent access is given in that
local gate: the missing theorem is still that factorized exact-softmax attention learns
the two required LEGO source edges and that the Transformer-local map implements the
group action. The transition table is a local reference operator, not a new
Transformer family.

Earlier controlled evidence remains subordinate:

- Phase II rejects a stable irreducible residual. Longer or scheduled training
  continues to improve. Dense composites reduce leakage, whereas rank-matched direct
  coordinates do not; this is evidence for a rank/function-class boundary, not a new
  low-rank theorem.
- Pythia-70M float64-v4 passes all eight checkpoint integrity audits. It is one
  pretraining trajectory. Its routing effects are nonmonotone and template-dependent,
  so it does not establish a population training law or the proposed
  diffuse-to-selective-to-sparse-collision narrative.
- The fixed-kernel clustering baseline verifies the prescribed-kernel depth dynamics
  only. It does not explain how training selects that kernel.

## Repository structure

| Layer | Main modules | Purpose |
|---|---|---|
| Data/model | data, model, controlled_model, model_variants | MQAR/LEGO laws and exact-softmax $QK/OV$ models |
| Training | controlled_training, phase2_study, population_gf | deterministic trajectories and exact population updates |
| Four gates | kernel_capacity, matrix_mqar, matrix_mqar_ode, mqar_matrix_gf, lego_single_step, lego_depth | exact matrix gradients/obstructions, adaptive ODE audit, capacity/access, local LEGO map, and depth composition |
| Theory | mqar_kernel_theory, metrics, interventions, phase2_analysis | closed equations, risk, transport, and edge blocking |
| External calibration | pretrained_bridge, pretrained_causal, pretrained_study, pretrained_analysis | audited Pythia checkpoint measurements |

REPOSITORY_SCOPE.toml is fail-closed: an unclassified module, configuration, report, or
result directory breaks the test suite. Historical exploratory work remains recoverable
at commit 1f06157.

## Install and verify

Python 3.11 is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[pretrained]"
PYTHONPATH=src python -m unittest discover -s tests -v
```

The CI also builds and imports a non-editable wheel.

Pythia calibration is cache-only unless network access is explicitly requested:

```bash
PYTHONPATH=src python -m routing_lab.pretrained_study \
  --config configs/pretrained_pythia70m_suite_a_calibration_float64_v4.json \
  --output-directory results/pretrained-pythia70m-suite-a-calibration-float64-v4 \
  --cache-directory /path/to/huggingface/cache
```

## Evidence rules

- Independent training seeds are the inferential units.
- Checkpoints, templates, layers, heads, clauses, and prompts are repeated measures.
- Accuracy alone does not identify an interaction kernel.
- A causal claim requires an explicit intervention and descendant recomputation.
- Failed numerical gates remain failures; thresholds are not changed after inspection.
- No model family or dataset is added before the active theorem gate closes.

The repository currently grants no third-party reuse license. Public visibility does
not grant permission to copy, modify, run, or redistribute the work.
