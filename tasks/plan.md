# Implementation plan: controlled routing dynamics to pretrained language models

## Overview

Phase I built the finite causal-Transformer laboratory and found residual cross-talk at
$d=8,C=32,m=4,H=4$. Phase II does **not** call that residual an open problem. It first
exhausts the six controls specified by the user, then carries the same function-level
and finite-intervention estimands into frozen pretrained causal language models.

The user already approved this matrix and requested autonomous execution. The terminal
choice is therefore: verify, independently review, then publish a curated snapshot.

## Assumptions frozen before implementation

1. The primary hard cell is attention-only $d=8,C=32,m=4,H=4,L=2$; the matching FFN
   cell is a replication and is never silently pooled with it.
2. The inferential unit is a seed. Paired cells reuse initialization, online data,
   evaluation episodes, and intervention streams wherever architecture permits.
3. The first pretrained bridge freezes weights. It asks whether the pretrained model
   performs token-level associative retrieval and where finite causal effects live.
4. The staged real-model ladder is 70M, 160M/410M, then 1B only after functionality,
   runtime, and storage gates.
5. The asymmetric QK estimand and episode-level per-slot $S_{\mathrm{key}}$ are
   implemented literally. The old midpoint split and target-edge-plus-attention screen
   remain exploratory.
6. Because $C=32>d=8$, an orthogonal hard-cell dictionary is impossible. Its coherence
   obeys the Welch lower bound
   $\mu\ge\sqrt{(C-d)/(d(C-1))}\approx0.311$. Orthogonal $E$ is therefore only a
   $C\le d$ negative control; the hard cell uses a deterministic near-Welch tight frame.

## Architecture decisions

- Preserve every v1 schema and hash contract. Phase-II uses a separate v2 control schema
  and model factory; adding defaults to the old `GridCell` would invalidate old hashes.
- Separate model adapters from estimands. Synthetic and Hugging Face models expose a
  small shared trace/intervention contract; architecture-specific hooks stay internal.
- A full $d\times d$ direct composite at $H=4,d_h=2$ has rank/capacity and parameter
  advantages over $Q_h^\top K_h$ or $O_hV_h$. It is only an upper-bound control.
  Optimization geometry is isolated with (a) an $H=1$ function-class-matched comparison
  and (b) a rank-$d_h$ gauge-fixed/manifold composite control.
- Keep raw observations long-form and immutable; all summaries/figures are derived.
- Use Pythia first because its official suite provides 154 checkpoints per size, the same
  data order across sizes, Apache-2.0 weights, and public training code.
- Model weights are a local cache, never a Git artifact. Manifests record model id,
  revision, hashes, tokenizer, dtype, device, and exact library versions.

## Task list

### Phase A — contracts and controlled architectures

#### Task 1: Freeze Phase-II estimands and v2 schemas

**Acceptance criteria**

- Define base risk, natural-swap MSE, Walsh leakage, late-time decay slopes/ratios,
  asymmetric QK contrast, finite module-suffix response, and causal per-slot
  $S_{\mathrm{key}}$.
- Give every cell a stable id, pairing policy, seeds, checkpoints, correction family,
  equivalence/practical thresholds, and decision label.
- Reject incomplete or confounded head/parameterization comparisons.

**Verification:** schema RED→GREEN tests and hand-computed mathematical cases.

**Files:** `reports/PHASE2_PROTOCOL.md`, `src/routing_lab/control_config.py`,
`tests/test_control_config.py`, `configs/phase2_*.json`.

#### Task 2: Direct-composite and embedding-source controls

**Acceptance criteria**

- Support factorized QK/OV, full direct composites (explicit capacity upper bound), and
  rank-matched gauge-fixed composites.
- Support learned, fixed normalized Gaussian, orthogonal when $C\le d$, and
  deterministic near-Welch low-coherence embeddings.
- At matched composites, forward scores, head updates, predictions, and traces agree.

**Verification:** analytic forward/gradient equivalence, rank/capacity audit,
serialization, and invalid-config tests.

**Dependencies:** Task 1.

**Files:** `src/routing_lab/model_variants.py`, `src/routing_lab/control_model.py`,
`tests/test_model_variants.py`.

#### Task 3: Horizon, scheduler, and genuine head-capacity controls

**Acceptance criteria**

- Constant LR and cosine decay continue exactly to 3200/6400 from saved optimizer and
  random-stream state.
- Compare (i) fixed residual width $d$, (ii) fixed per-head width $d_h$, and (iii) fixed
  total parameter budget using variable attention inner width plus an explicitly audited
  budget-balancing module. Standard MHA's fixed-$d$ and attention-parameter matching are
  recognized as the same comparison, not counted twice.
- Every comparison reports $d,H,d_h,p=Hd_h$, attention parameters, total parameters,
  and which quantities remain unmatched.

**Verification:** replay/resume equality, scheduler values, and parameter-count tests.

**Dependencies:** Tasks 1–2.

**Files:** `src/routing_lab/control_training.py`, `src/routing_lab/control_run.py`,
`tests/test_control_training.py`, `configs/phase2_training_limits.json`.

#### Checkpoint A

- Legacy plus new contract tests pass.
- Four calibration seeds freeze only runtime/numerical choices, never favorable outcomes.

### Phase B — finite mechanisms and population gradient

#### Task 4: Registered finite causal localization

**Acceptance criteria**

- Block every memory slot separately and compute
  $\delta_i=(\hat y-\hat y^{(-i)})y$ and
  $S_{\mathrm{key}}=\mathbb E[\delta_J-(m-1)^{-1}\sum_{i\ne J}\delta_i]$.
- For QK, OV, FFN, and readout, use the same on-manifold swap and compute the
  site-specific finite suffix response $G_{M,e}(z+\Delta)-G_{M,e}(z)$.
- Never use adjacent coherent donor-patch equivalence as attenuation evidence; tangent
  results are only local approximations to a measured finite effect.

**Verification:** exact mask cases, finite identities, and small-$\epsilon$ tangent limits.

**Dependencies:** Tasks 1–3.

**Files:** `src/routing_lab/finite_localization.py`,
`src/routing_lab/interventions.py`, `tests/test_finite_localization.py`.

#### Task 5: Exact population GF-like bridge

**Acceptance criteria**

- Enumerate ordered distinct concept tuples, target slots, and all $2^m$ values for
  registered small $(C,m)$ cells with exact multiplicity.
- Run full-batch small-step GD and report the same order parameters as online AdamW/SGD.
- Verify population risk, gradient, and step-halving convergence independently.

**Verification:** exact row count/risk, finite-difference gradient, and step-halving tests.

**Dependencies:** Tasks 1–2.

**Files:** `src/routing_lab/population_gf.py`, `tests/test_population_gf.py`,
`configs/phase2_population_gf.json`.

#### Checkpoint B

- Full tests pass; outputs have unique keys, finite values, Parseval checks, and hashes.

### Phase C — production toy matrix and analysis

#### Task 6: Run the six-axis matrix

**Acceptance criteria**

- Frozen production configs use at least 10 paired seeds for primary contrasts.
- Failures remain in a ledger and trigger literature-backed remedies before promotion.
- Checkpoints and manifests are resumable, immutable, and independently replayable.

**Verification:** completeness, CRN identity, hashes, duplicate audit, representative replay.

**Dependencies:** Checkpoints A–B.

**Files:** `configs/phase2_*.json`, `results/toy-controlled-matrix-v1/`,
`autoresearch/orchestrator-260820-1726/`.

#### Task 7: Trajectory statistics and mechanism classification

**Acceptance criteria**

- Compare base risk and leakage with paired late-time log slopes, leakage/risk ratios,
  floor-aware censoring, and prespecified sensitivity windows.
- Use whole-seed resampling and family correction for confirmatory endpoints; label all
  selected slices exploratory.
- Show seed distributions, trajectories, uncertainty, module/site effects, head geometry,
  and failures—not only means.

**Verification:** independent numerical recomputation and rendered PNG/SVG inspection.

**Dependencies:** Task 6.

**Files:** `src/routing_lab/control_analysis.py`,
`src/routing_lab/control_figures.py`, `tests/test_control_analysis.py`,
`results/toy-controlled-analysis-v1/`.

### Phase D — pretrained causal-LM bridge

#### Task 8: Versioned GPT-NeoX/Pythia adapter

**Acceptance criteria**

- Record hidden states, attention probabilities, per-head value/output updates, logits,
  and exact token positions without changing logits in observation-only mode.
- Support deterministic target/distractor edge masks and finite residual/head
  interventions with descendants recomputed.
- Save exact model revision, tokenizer/prompt tokens, dtype, device, and packages.

**Verification:** no-op logit equivalence, mask semantics, hook cleanup, deterministic reload.

**Dependencies:** Task 1 and official Transformers documentation.

**Files:** `src/routing_lab/pretrained_adapter.py`,
`src/routing_lab/pretrained_tasks.py`, `tests/test_pretrained_adapter.py`.

#### Task 9: Pythia training-time and scale pilots

**Acceptance criteria**

- Use held-out collision-free symbolic mappings and balanced labels.
- Evaluate at least four training revisions and two sizes; proceed upward only after a
  functional gate or a powered negative capability result.
- Reuse episode ids across checkpoints/sizes for function, causal routing, geometry, and
  finite site effects.

**Verification:** revision/hash and token audits, repeated-load equality, paired inference.

**Dependencies:** Task 8.

**Files:** `configs/pretrained_pythia_pilot.json`,
`src/routing_lab/pretrained_run.py`, `tests/test_pretrained_run.py`,
`results/pretrained-pythia-pilot-v1/`.

#### Checkpoint D

- A frozen pretrained family either passes retrieval functionality with causal measures,
  or yields a well-powered negative boundary; a nonfunctional task is not interpreted.

### Phase E — review and publication

#### Task 10: Independent validation and clean GitHub snapshot

**Acceptance criteria**

- Recompute every headline number; every figure has a chart contract and uncertainty.
- Independent review finds no Critical/Required issue in math, statistics, causal wording,
  code, provenance, or public scope.
- Publish commented code, configs, verified aggregates, reports, and commands; exclude
  caches and unnecessary raw checkpoints.

**Verification:** clean-clone full tests, hash/path/secret audit, and remote tree equality.

**Dependencies:** Checkpoint D.

**Files:** `reports/PHASE2_RESULTS.md`, `reports/PHASE2_VALIDATION.json`,
`README.md`, curated result directories.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---:|---|
| Home has only 14GB free | High | Stage 70M first, set a cache budget, delete nothing without approval. |
| Small LMs fail symbolic retrieval | High | Tune prompt only on a development split, freeze it, then evaluate held-out episodes. |
| Head controls are confounded | High | Encode design mode and report all dimension/parameter deltas. |
| Direct composites change capacity | High | Separate rank-matched/H=1 geometry controls from a full-rank upper bound. |
| Diagnostics invite selective reporting | High | Freeze endpoint families and corrections before production. |
| Hooks change model behavior | High | Require no-op logit equality and automatic cleanup tests. |
| Residual called open too early | High | Require persistence across every registered remedy and independent confirmation. |

## Official implementation sources

- https://huggingface.co/EleutherAI/pythia-70m-deduped
- https://github.com/EleutherAI/pythia
- https://huggingface.co/docs/transformers/model_doc/gpt_neox
