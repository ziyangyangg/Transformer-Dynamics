# Implementation Plan: Learning Task-Aligned Interaction Kernels

This is the only active plan. Historical toy-to-Pythia plans remain in Git history and
immutable result manifests.

## Success criterion

The project must close the chain

$$
(\mathcal D,R,\theta_0)
\xrightarrow{\text{population gradient flow}}
\{B_{\ell h}(s),C_{\ell h}(s)\}
\xrightarrow{\text{exact softmax}}
\mathcal K_{\ell,s}
\xrightarrow{\text{network depth}}
\Phi_{\theta_s}^{L}(X).
$$

A valid result must identify which statistics of the task distribution drive the
factorized parameter updates, prove when the correct source margin and value transport
emerge, and propagate that learned-kernel error through finite network depth. Adding
heads, layers, datasets, or parameters is not completion.

The immediate deliverable is a condition-discovery theorem. It must name measurable
quantities for task identifiability $\kappa$, representability
$\varepsilon_{\rm cap}$, and factor access.
Use $(\mu_B,\mu_C)$ only as one sufficient certificate for factor access.
Every additional constant must be derived from the chosen data and model, and each
gate must have an exact failure witness. Each candidate condition must cite its
data/model source, exact witness, experiment, and prior-theory boundary before it is
generalized. The conditions may restrict a broad task class, but they may not assume
the desired kernel.

## Gate 0: scope and prior art

- [x] Locate the training gap in Section 10 of the current arXiv version of
  *A Mathematical Perspective on Transformers*.
- [x] Separate fixed-kernel depth dynamics from parameter-training dynamics.
- [x] Record prior results on max-margin token selection, co-occurrence gradient flow,
  induction heads, multi-head ICL, and LEGO state tracking.
- [x] Treat retrieval, rank, collisions, and patching as subordinate diagnostics.

A novelty statement must exceed both a training-only special case and a fixed-kernel
dynamics result.

## Repository gate

- [x] Retain only the theorem-facing model, training, measurement, and audited evidence
  dependency closure.
- [x] Remove exploratory scaling, landscape, localization, and report-builder branches
  from the active tree.
- [x] Enforce the allowlist in REPOSITORY_SCOPE.toml.

The pre-cleanup state remains recoverable at commit 1f06157.

## Gate 1: exact MQAR kernel learning

The data are the single-query, binary-value specialization of MQAR. The first model is
one layer, one head, exact softmax, factorized Q/K/O/V, a trained readout, and no FFN.

- [x] Derive the closed population risk for the permutation-symmetric parameterization.
- [x] Verify the equations against the complete value cube and automatic
  differentiation.
- [x] Prove that positive nondegenerate factors satisfying the alignment condition
  yield

$$
\delta(s)\to+\infty,
\qquad
g(s)\to1,
\qquad
\mathcal E_{\mathcal K}(s)=2R(s)\to0.
$$

- [x] Refute the unrestricted initialization claim with the exact Q=K=0 factorization
  barrier.
- [x] Retain the signed-gain exact-softmax counterexample to internal route
  identifiability.
- [ ] Lift the scalar proof to learned matrix-valued embedding directions under the
  registered $\kappa,\varepsilon_{\rm cap},\mu_B,\mu_C$ gates, or produce an exact
  counterexample to that condition set.

The positive theorem and both obstructions are stated in
reports/MQAR_KERNEL_LEARNING_THEOREM.md. Gate 1 is resolved for the registered
reduced parameterization, not for a general Transformer.

## Gate 2: LEGO training-to-depth bridge

The next and only active extension changes the data, not the model family. It uses the
published LEGO law:

$$
x_t=g_t(x_{t-1}),
\qquad
y_t=g_t(y_{t-1}),
$$

with variables sampled without replacement, initial state uniform, and actions sampled
with replacement.

- [x] Implement the complete finite cyclic-group population and five-token clause
  encoding.
- [x] Register the two required source clauses for each transition: the current
  predicate and the previous answer.
- [ ] Derive the factorized population gradient equations for one LEGO transition
  using the same condition quantities as MQAR.
- [ ] Prove or refute a per-step task-weighted kernel error bound with explicit
  $\varepsilon_{\rm cap}$.
- [ ] Compose the per-step bound through a chain of length L.
- [ ] Separate the new training-to-depth statement from the existing LEGO
  learnability and length-generalization theorem.

Gate 2 passes only when the same theorem derives a kernel condition from gradient flow
and uses that condition to bound finite-depth state-tracking error.

## Later architecture changes

No architecture change is active. A component may be added only when Gate 2 exposes a
specific failed assumption:

1. multiple heads, only for signed allocation or cancellation;
2. residual depth, only for operator composition and identifiability;
3. RMSNorm or FFN, only if it changes a proved bound;
4. finite rank, only if an explicit capacity term enters the theorem.

Each change must reuse the same data law, metrics, paired initialization, and
independent training seeds.

## Stop rules

- Do not tune one C=32, d=8 cell indefinitely.
- Do not present low rank, nonorthogonal embeddings, or fixed-QKV clustering as new.
- Do not infer a kernel from accuracy or an attention map.
- Do not treat checkpoints, templates, layers, heads, or prompts as independent
  samples.
- Do not add another model family or dataset before the LEGO theorem either closes or
  fails with an exact counterexample.
