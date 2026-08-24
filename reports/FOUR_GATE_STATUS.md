# MQAR-to-LEGO: Four-Gate Status

## Fixed question

For an exact-softmax Transformer trained by population gradient flow, when do the
factorized composites

$$
B_h=Q_h^{\top}K_h,
\qquad
C_h=O_hV_h
$$

learn the messages required by the task, and how does the resulting local error
accumulate through depth?

## What is implemented

**1. MQAR capacity and access.** On the complete value cube,

$$
\mathcal E_{\mathcal K}^2
=
(\widehat f_{\{J\}}-1)^2
+\sum_{S\ne\{J\}}\widehat f_S^2
=2R.
$$

The code computes this identity exactly. It also compares the direct composite
gradient with the factor velocity

$$
\dot B=-G_BK^{\top}K-Q^{\top}QG_B,
\qquad
\dot C=-G_CV^{\top}V-OO^{\top}G_C.
$$

The dense arm is only a constructive capacity upper bound. The rank-matched arm is
an optimization-coordinate control. At $Q=K=0$, $G_B$ can be nonzero while
$\dot B=0$; this is an exact access obstruction.

**2. Full-matrix population GF.** All arms start from the same function and use the
complete MQAR population of size

$$
|\Omega|=(C)_m\,m\,2^m.
$$

Explicit Euler trajectories are compared on one physical-time grid under step
halving. This is a deterministic numerical experiment, not a matrix-valued
convergence theorem.

**3. LEGO single step.** From the published cyclic LEGO law, the local target is

$$
y_{t+1}=y_t+g_{t+1}\pmod k.
$$

The complete $k^2$ parent-pair population trains a stochastic transition matrix
$P_\theta(g)$ by exact population cross-entropy. Parent access is given; attention
routing is not trained in this gate. This table is a local reference operator: it is
not evidence that a Transformer FFN or readout learns the same operation from clauses.

**4. Depth composition.** For learned local matrices $A_t$ and target matrices $B_t$,
the code verifies the exact telescope

$$
A_L\cdots A_1-B_L\cdots B_1
=
\sum_{t=1}^{L}
A_L\cdots A_{t+1}(A_t-B_t)B_{t-1}\cdots B_1,
$$

and therefore

$$
\|\widehat p_L-p_L^*\|_2
\le
\sum_{t=1}^{L}
\|A_t-B_t\|_2
\prod_{r=t+1}^{L}\|A_r\|_2.
$$

The bound is exhaustively checked over every initial state and action string up to a
chosen depth.

## Exact boundary

The four implementations separate capacity, factor access, local computation, and
error composition. They do **not** yet prove that a factorized Transformer trained on
LEGO clauses learns the two required source edges or the local group action. The
remaining theorem is precisely:

$$
\text{population GF in the Transformer parameters}
\Longrightarrow
\eta_{\rm route}(s)+\eta_{\rm local}(s)\to0,
$$

under explicit task-identifiability, finite-capacity, and factor-access conditions;
then insert both errors into the depth bound above.
This is the active training-aware Transformer-dynamics problem.

## Reproduction

```bash
PYTHONPATH=src python -m unittest \
  tests.test_kernel_capacity \
  tests.test_mqar_matrix_gf \
  tests.test_lego_single_step \
  tests.test_lego_depth -v
```

The implementations are respectively `kernel_capacity.py`, `mqar_matrix_gf.py`,
`lego_single_step.py`, and `lego_depth.py`.
