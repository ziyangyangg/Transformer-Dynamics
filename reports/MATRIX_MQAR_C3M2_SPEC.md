# Specification: Minimal Matrix MQAR Kernel Learning

## Objective

Freeze the smallest finite problem that distinguishes matrix-valued kernel learning
from the existing radial theorem.  The deliverable is either a correct convergence
argument or an exact non-task-aligned critical point.  Numerical trajectories are
admissible only after an independent adaptive-ODE tolerance audit.

## Data law

Let the concept set be $[C]=\{0,1,2\}$ and let $m=2$.  The complete population is

$$
\Omega
=
\left\{
(c_1,c_2,J,v_1,v_2):
c_1\ne c_2,\ J\in\{1,2\},\ v_i\in\{-1,+1\}
\right\},
$$

with the uniform law.  Thus $|\Omega|=(3)_2\,2\,2^2=48$.  The query concept and
label are

$$
q=c_J,
\qquad
Y=v_J.
$$

This is an exact single-query MQAR population.  It is a proof object, not a claim
about natural-language frequencies.

## Minimal model

Use $d=3$, one layer, one head, exact softmax, no FFN, no normalization, and no
residual value bypass.  Let

$$
E\in\mathbb R^{3\times3},
\quad
B=Q^\top K\in\mathbb R^{3\times3},
\quad
C=OV\in\mathbb R^{3\times3},
\quad
w,u\in\mathbb R^3,
$$

where $u=(1,0,0)^\top$ is fixed and $e_c=E_{c,:}^\top$ is learned.  Concept
vectors enter the score path, while the signed scalar values enter the value path:

$$
s_i=e_q^\top B e_{c_i},
\qquad
a_i=\frac{e^{s_i}}{1+e^{s_1}+e^{s_2}},
$$

$$
g=w^\top C u,
\qquad
f_\theta(X)=g\sum_{i=1}^2 a_i v_i.
$$

The additional denominator term is a causal query-self position with fixed score
zero and value zero.  The role separation is the first registered assumption to be
tested later; it does not fix the learned score matrix or the attention weights.

## Gauge-equivalent parameters and target kernel

The prediction depends only on

$$
S=EBE^\top,
\qquad
g=w^\top C u.
$$

The following transformations preserve the function:

$$
Q\mapsto AQ,
\quad
K\mapsto A^{-\top}K;
\qquad
O\mapsto OH,
\quad
V\mapsto H^{-1}V,
$$

$$
E\mapsto ET^\top,
\quad
B\mapsto T^{-\top}BT^{-1};
\qquad
C\mapsto \alpha C,
\quad
w\mapsto \alpha^{-1}w,
$$

for invertible $A,H,T$ and nonzero $\alpha$.  Raw factor norms are therefore not
the mathematical target.

For slot $i$, define the delivered value coefficient

$$
\mathcal K_\theta(X;i)=g a_i(X),
$$

and the task kernel

$$
\mathcal K^*(X;i)=\mathbf 1\{i=J\}.
$$

Because scores are value-blind and the values are independent Rademacher signs,

$$
\mathcal E_{\mathcal K}(\theta)
=
\mathbb E_\Omega
\sum_{i=1}^2
\left(g a_i-\mathbf 1\{i=J\}\right)^2
=2R(\theta).
$$

## Required symbolic objects

For $r=\sum_i a_i v_i$, $\epsilon=gr-Y$, and

$$
\lambda_i=\epsilon g a_i(v_i-r),
\qquad
\gamma=\mathbb E_\Omega[\epsilon r],
$$

the implementation must verify

$$
G_B
=
\mathbb E_\Omega\sum_i\lambda_i e_qe_{c_i}^\top,
$$

$$
G_{e_c}
=
\mathbb E_\Omega\left[
\mathbf 1\{q=c\}\sum_i\lambda_iBe_{c_i}
+
\sum_i\mathbf 1\{c_i=c\}\lambda_iB^\top e_q
\right],
$$

$$
G_C=\gamma wu^\top,
\qquad
G_w=\gamma Cu.
$$

The factorized flow must use the exact chain rule

$$
\dot Q=-KG_B^\top,
\quad
\dot K=-QG_B,
\quad
\dot O=-G_CV^\top,
\quad
\dot V=-O^\top G_C,
\quad
\dot E=-G_E,
\quad
\dot w=-G_w.
$$

## Success criteria

1. Complete-population enumeration has exactly 48 uniformly weighted episodes.
2. Hand gradients agree with automatic differentiation in float64.
3. Prediction, risk, and target-kernel error are invariant under every registered
   gauge transformation.
4. The finite quotient $(S,g)$ is classified: either its critical points are solved,
   or a checkable counterexample is produced.
5. Every non-task-aligned parameter critical family records the access degeneracy
   that creates it.
6. A nondegenerate trajectory is integrated by float64 adaptive DOP853 twice with
   independently frozen tolerances; risk, quotient observables, and invariant drifts
   must agree.  Failure forbids numerical interpretation.

## Boundaries

- Always: use the complete population, float64 reductions, exact softmax, and
  gauge-invariant observables.
- Ask first: change $C,m,d$, add heads/layers/FFNs, or change the data law.
- Never: infer a theorem from a trajectory; call dense direct training a pure
  optimization control; or hide a failed tolerance audit.
