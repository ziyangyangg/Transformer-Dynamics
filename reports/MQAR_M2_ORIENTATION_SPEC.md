# MQAR M2: signed Q/K orientation experiment

## Objective

M1 established that exact zero Q/K factors form an invariant access barrier. M2
asks the next, narrower question: after fixing nonzero access and every non-QK
parameter, does the relative sign of the initial Q/K factors change whether training
learns a task-aligned score kernel?

For layer $\ell$ and head $h$, write

$$
B_{\ell h}=Q_{\ell h}^{\top}K_{\ell h}.
$$

The paired interventions set

$$
K_{\ell h}(0)=\sigma Q_{\ell h}(0),
\qquad \sigma\in\{+1,-1\},
$$

at Q/K scales $r\in\{1,2^{-8}\}$. Thus the two signs have identical Q/K
norms and Gram matrices and share the same embeddings, OV/FFN/readout parameters,
optimizer, and counter-addressed MQAR batches. An independently sampled standard
Q/K arm is a descriptive natural-initialization reference.

The sign intervention is not itself called correct routing. Task alignment is
measured on data by

$$
\Delta_{\ell h}(t)=\mathbb E\left[
s_{\ell h}(q,k_{\mathrm{target}})
-\frac1{m-1}\sum_{j\ne J}s_{\ell h}(q,k_j)
\right].
$$

## Frozen design

- Data: the public Zoology-compatible MQAR law already audited in M1.
- Model: the unchanged M1 Transformer: four layers, $d=128$, four heads,
  FFN width 512, pre-RMSNorm, RoPE, tied embeddings.
- Training: AdamW, learning rate $10^{-3}$, zero weight decay, 6,400 steps.
- Independent units: seeds 100--119.
- Arms: `independent`, `positive`, `negative`, `positive-small`,
  `negative-small`.
- Checkpoints: 0, 200, 400, 800, 1,600, 3,200, 6,400.
- Primary population: $(L,m)=(256,16)$ for endpoint accuracy.
- Routing population: $(L,m)=(64,4)$ for $\Delta_{\ell h}$.

All arm comparisons are paired within the training seed. Layers, heads,
checkpoints, populations, and examples are repeated measurements, never additional
independent samples.

## Estimands and decision boundary

For each scale, the primary contrast is

$$
D_{\mathrm{acc}}(r)
=\mathbb E_{\mathrm{seed}}[
\mathrm{Acc}_{+,r}(6400)-\mathrm{Acc}_{-,r}(6400)].
$$

The mechanism contrast is the seed-level difference in the largest head score
margin,

$$
D_{\Delta}(r)=\mathbb E_{\mathrm{seed}}[
\max_{\ell,h}\Delta^{+}_{\ell h}(6400)
-\max_{\ell,h}\Delta^{-}_{\ell h}(6400)].
$$

The four contrasts use one 20,000-resample whole-seed max-$T$ family. M2 reports:

1. **signed separation** only when both simultaneous lower bounds at a scale are
   positive;
2. **persistent finite-horizon failure candidate** only when the negative arm has
   mean final accuracy below 0.5 and mean improvement from step 3,200 to 6,400 below
   0.05;
3. **architectural repair** when the negative arm reaches mean final accuracy at
   least 0.8;
4. otherwise **unresolved finite-horizon behavior**.

These labels are experimental classifications, not gradient-flow theorems.

## Measurements

Each checkpoint records accuracy, NLL, Q/K norm, Q/K gradient norm, per-layer/head
Q/K cosine, normalized composite trace, composite skew fraction, target and
distractor key attention, and $\Delta_{\ell h}$. Step zero must verify:

$$
\cos_F(Q,K)=\sigma,
\qquad
B_{\ell h}=\sigma Q_{\ell h}^{\top}Q_{\ell h},
$$

to numerical precision, while all non-QK parameters are bitwise identical across
paired arms.

## Commands

```bash
PYTHONPATH=src python -m routing_lab.mqar_m2_study \
  --config configs/mqar_m2_orientation_v1.json \
  --output-directory results/mqar-m2-orientation-v1 \
  --device cuda

PYTHONPATH=src python -m routing_lab.mqar_m2_analysis \
  --source-directory results/mqar-m2-orientation-v1 \
  --output-directory results/mqar-m2-orientation-v1-analysis \
  --report-path reports/MQAR_M2_ORIENTATION_RESULT.md
```

## Project structure and code style

- `mqar_m2.py`: initialization and measurement primitives.
- `mqar_m2_study.py`: config, resumable execution, and strict raw-artifact audit.
- `mqar_m2_analysis.py`: seed-grain inference, figures, and concise report.
- `test_mqar_m2*.py`: behavior and tamper-resistance tests.

Public functions use frozen dataclasses, explicit tensor shapes, isolated random
generators, and comments that explain scientific intent rather than restating code.
No new dependency is allowed.

## Success criteria

- RED tests prove sign/magnitude pairing, non-QK identity, complete checkpoint grids,
  raw-to-aggregate reconstruction, seed-grain inference, and byte determinism.
- A small CPU/GPU smoke passes before production.
- Exactly 100 production trajectories complete without non-finite values.
- Every result is source/config/environment hash-bound and independently reloadable.
- The report states whether the reduced negative branch transfers, is repaired, or
  remains unresolved; it does not relabel finite AdamW evidence as a theorem.
- Full tests, Ruff, wheel installation, GitHub push, and GitHub Actions pass.

## Boundaries

- Always: reuse the M1 data/model law and pair every arm within seed.
- Ask first: changing optimizer, task law, architecture, or inferential unit.
- Never: select arms after seeing outcomes; count heads/templates as independent;
  claim the signed initialization is a necessary or sufficient condition; enter
  LEGO, GPT-2, or another dataset in M2.
