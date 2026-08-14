# Specification: causal routing and compressed-concept compensation lab

## Objective

Build a self-contained, readable PyTorch laboratory that distinguishes three claims:

1. low population risk forces end-to-end causal dependence on the queried value;
2. training selects a particular factorization through learned embeddings, QK, OV, FFN,
   and readout;
3. a learned compressed concept code creates an intermediate distractor effect that a
   downstream module suppresses.

The implementation must never infer claim 2 or 3 from accuracy alone.  Each claim has a
separate estimand and validation criterion below.

## Probability model

Let `C` be the concept vocabulary size and `m` the number of memory slots.  A sample is

```
(c_1, v_1), ..., (c_m, v_m), q, y
```

with ordered distinct concepts sampled uniformly without replacement,
`v_i iid Uniform({-1,+1})`, `J ~ Uniform({1,...,m})`, `q=c_J`, and `y=v_J`.
Fresh values are sampled every episode, so `P(y=1 | q)=1/2`; an embedding cannot store
the answer.

An on-support distractor swap chooses `K != J` and a concept `c_new` absent from the
memory, then replaces only `c_K` by `c_new`.  Values, target index, query, and label are
unchanged.  Both endpoints have positive probability under the original distribution.

## Network

The model is a finite pre-RMSNorm causal Transformer.  Its input states are

```
x_i^0 = E[c_i] + v_i e_value + p_i + e_memory
x_T^0 = E[q] + p_T + e_query
```

For layer `ell` and head `h`, with `d_h=d/H`, define

```
z_t = RMSNorm(x_t)
B_ell,h = Q_ell,h^T K_ell,h
C_ell,h = O_ell,h V_ell,h
s_ti,h = beta / sqrt(d_h) * z_t^T B_ell,h z_i
a_ti,h = softmax_{i<=t}(s_ti,h)
u_t,h = sum_{i<=t} a_ti,h z_i
x_t^att = x_t + L^(-1/2) sum_h C_ell,h u_t,h
x_t^(ell+1) = x_t^att + L^(-1/2) FFN_ell(RMSNorm(x_t^att))
```

The FFN is optional and equals `U2 GELU(U1 z + b1) + b2`.  The scalar prediction is
`f_theta(x)=w^T RMSNorm(x_T^L)+b`.

## Loss and training

The theoretical objective is the population squared loss

```
R(theta) = 1/2 E[(f_theta(X)-Y)^2],    d theta / ds = -grad R(theta).
```

Experiments approximate the expectation with fresh online batches.  AdamW uses zero
weight decay.  A tuned momentum-SGD control is treated as a discrete approximation to
gradient flow; optimizer agreement is a robustness check, not an identity of dynamics.

## Causal estimands

The end-to-end value kernel for slot `i` is the controlled finite difference

```
kappa_i(c, v_-i, J) = 1/2 [f(do(v_i=+1)) - f(do(v_i=-1))].
```

Its average over the other random values is the first-order Walsh coefficient.  The
registered value-flip statistic is

```
Xi_value = 1/2 E[Y * (f(X)-f(do(v_J=-v_J)))].
```

The direct target-key path statistic sets the query-to-target logit to `-inf` in every
layer and head, renormalizes softmax, and recomputes descendants:

```
Xi_key = E[Y * (f(X)-f(do(s_query,J^(ell,h)=-inf for all ell,h)))].
```

`Xi_key` is a path-specific internal intervention.  It is not the total causal effect:
the target value may first flow to a later memory token and reach the query indirectly.

## Superposition terminology

This lab distinguishes:

- compressed dictionary geometry: `C>d`, so the concept vectors cannot all be
  orthogonal;
- activation superposition: one hidden state simultaneously contains multiple
  independently decodable concept/value features.

The present embedding experiments directly test only compressed dictionary geometry.
Calling a low effective rank "activation superposition" is forbidden without a decoder
or simultaneous-feature test.

For normalized rows `u_c=E_c/||E_c||`, record

```
Gram(c,c') = u_c^T u_c'
mu(E) = max_{c!=c'} |Gram(c,c')|
r_eff(E) = (sum sigma_i^2)^2 / sum sigma_i^4
D_c = ||E_c||^2 / sum_c' (u_c^T E_c')^2
```

and verify `sum_c D_c <= rank(E) <= d`.  Geometric overlap alone is not functional
cross-talk.

## On-support cross-talk and compensation

For a base example `X` and valid distractor-swapped example `X'`, let `Z_r(X)` be a
registered internal site.  Define the hybrid patch output

```
f_r^patch(X,X') = f(X; do(Z_r = Z_r(X')))
I_r = E[(f_r^patch(X,X')-f(X))^2].
```

Registered sites are query-row QK scores, query-row attention probabilities, per-head
pre-OV mixtures, per-head post-OV updates, post-attention query residual, FFN branch,
post-FFN query residual, and final prediction.  Descendants are recomputed after a patch.

For consecutive sites around module `M`, define the output-unit suppression contrast

```
C_M = log(I_before(M)+1e-12) - log(I_after(M)+1e-12).
```

A module is called a compensator only if all of the following hold on held-out seeds:

1. the swap is on support and the label is unchanged;
2. `I_before(M)` is practically nonzero under a preregistered floor;
3. the seed-level confidence interval for `C_M` excludes zero in the suppressing
   direction;
4. the conclusion replicates under a second optimizer or architecture control;
5. overall accuracy and `Xi_value` remain matched.

## Tangent decomposition

The chord direction for a distractor swap is `delta e=E[c_new]-E[c_old]`.  Interior
points of this chord are not data-distribution samples, so the tangent is a local
mechanistic diagnostic, not an on-support causal estimand.

For one query attention head,

```
U(z) = C sum_i a_i(z;B) z_i,  m = sum_i a_i z_i
```

the exact first-order input perturbation is decomposed as

```
D U[delta z]
 = C sum_i a_i delta z_i
 + C sum_i a_i (z_i-m) delta s_i,

delta s_i = beta/sqrt(d_h) *
  [(delta z_q)^T B z_i + z_q^T B delta z_i].
```

The first term is the value/content path; the second is the QK/softmax route path.  The
sum must match autograd JVP and centered finite differences.  For a residual branch
`z_plus=z+F(z)`, use the downstream adjoint `r=grad_{z_plus} f` and record

```
t_skip   = r^T delta z
t_branch = r^T J_F delta z
t_total  = t_skip + t_branch.
```

Opposite signs show cancellation in the output-relevant direction.  Merely propagating
`grad f dot delta z` from layer to layer cannot localize compensation because the chain
rule makes that total derivative invariant.

## Statistical design

Primary grid:

- `d=16`, `L=2`, `m=4`;
- `C in {16,64}` and `H in {1,4}`;
- attention-only and FFN width `2d`;
- AdamW and tuned momentum SGD;
- at least 10 independent seeds per primary cell that passes the accuracy gate.

Scaling controls use `d in {8,16,32}`, `C/d in {1,2,4}`, and `m in {2,4,8}` when compute
permits.  Evaluation uses at least 8192 fresh examples per checkpoint, in independent
chunks.

The independent inferential unit is the training seed.  Within-seed examples reduce
Monte Carlo error but do not increase the seed-level sample size.  Report means,
standard deviations, paired contrasts, percentile bootstrap 95% intervals, standardized
paired effects, and every seed value.  The primary head-by-load interaction is

```
I_rank = [r_eff(C=64,H=4)-r_eff(C=64,H=1)]
       - [r_eff(C=16,H=4)-r_eff(C=16,H=1)].
```

Intervention endpoints and module-localization contrasts are fixed before examining
their direction.  Failed seeds, NaNs, accuracy-gate failures, and optimizer remedies are
retained in a failure ledger.

## Commands

All commands use the existing environment:

```
PY=/home/zion/miniforge3/envs/llm4rec/bin/python3.11
Test:     PYTHONPATH=src $PY -m unittest discover -s tests -v
Smoke:    PYTHONPATH=src $PY -m routing_lab.run --config configs/smoke.json
Primary:  PYTHONPATH=src $PY -m routing_lab.run --config configs/primary.json
Analyze:  PYTHONPATH=src $PY -m routing_lab.analyze --run-dir results/primary
```

## Project structure

```
src/routing_lab/  model, data, interventions, training, statistics
tests/            deterministic unit and integration tests
configs/          immutable experiment grids
results/          raw per-seed records, checkpoints, analyses, figures
reports/          mathematical and empirical interpretation
autoresearch/     iteration ledger and keep/discard decisions
tasks/            implementation plan and task status
```

## Code style

Use typed dataclasses, explicit tensor shapes in docstrings, small pure functions, and
comments that explain mathematical meaning rather than restating syntax.  No hidden
global RNG and no silent metric fallback.

## Testing strategy

- Unit: distribution support, label invariance, Fourier/causal identities, geometry
  inequalities, tangent decomposition, bootstrap determinism.
- Integration: trace replay and each patch site reproduce an explicit recomputation;
  finite-difference/JVP agreement; checkpoint round trip.
- Experiment smoke: every registered metric is finite and serialized with its exact
  configuration and seed.

## Boundaries

- Always: fresh held-out samples; seed-level inference; save raw observations; retain
  negative results; validate all interventions.
- Ask first: add dependencies, change the data-generating process, publish or push.
- Never: edit `../sources`; tune an intervention after seeing its desired direction;
  call rank alone superposition; call correlation causal.

## Success criteria

1. All tests pass, including finite-difference/JVP and on-support checks.
2. At least 10 successful independent seeds per primary cell or a documented compute/
   optimization failure with remedy grid.
3. Raw per-seed metrics, checkpoint metadata, analysis tables, and figures are produced.
4. Every empirical claim is classified as established, candidate, falsified, or
   unidentified.
5. The final report states the exact model, formulas, statistical estimands, observed
   numbers, literature boundary, and theorem/counterexample targets.

