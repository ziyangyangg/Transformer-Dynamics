# Single-layer kernel selection: current theorem boundary

## Exact object

For the complete $C=d=3$, $m=2$ matrix-MQAR population, let

$$
S=EQ^{\top}KE^{\top},
\qquad
g=w^{\top}Oz,
$$

and let $a_{qd},b_{qd}$ be the exact-softmax target and distractor masses. The
delivered kernel and target kernel are

$$
\kappa(\theta)=\bigl((ga_{qd},gb_{qd})\bigr)_{q\ne d},
\qquad
\kappa^*=\bigl((1,0)\bigr)_{q\ne d}.
$$

Complete value enumeration gives the identity

$$
R(\theta)=\frac1{12}\|\kappa(\theta)-\kappa^*\|_2^2.
$$

Thus risk convergence and delivered-kernel convergence are exactly equivalent in
this population. This is the smallest rigorous training-side object that can later
feed a fixed-kernel depth-dynamics theory.

## What is proved

On the role-tied one-head reduction, positive factors and the initial inequality
$h(g_0,\delta_0)<1$, whose region is forward invariant, imply

$$
\delta(s)\to+\infty,
\qquad
g(s)\to1,
\qquad
R(s)\to0.
$$

For the unrestricted matrix factorization, the quotient risk has no finite critical
point, but raw parameter gradient flow can lose access to a nonzero quotient gradient.
Two externally verified counterexamples settle the previous broad conjectures:

1. an ambient-open large-norm basin approaches a self-only kernel with $R\to1/2$;
2. a scale-uniform open cone of arbitrarily small initializations approaches
   $ga_{qd}=gb_{qd}=1/2$ and $R\to1/4$.

The second basin has positive lower Lebesgue density at the origin. Therefore neither
“almost every regular initialization succeeds” nor “failure density vanishes under
small initialization” is true. Full rank, nonzero initial access, exact balance, and
small norm are insufficient because none fixes the task orientation of $Q^{\top}K$.

The exact identities behind the classification are

$$
I_{QK}=QQ^{\top}-KK^{\top},
\qquad
I_{EQK}=E^{\top}E-Q^{\top}Q-K^{\top}K,
$$

$$
J_L=ww^{\top}-OO^{\top},
\qquad
J_R=O^{\top}O-zz^{\top},
$$

and

$$
\dot R=-\frac1{36}\|J_\kappa(\theta)^{\top}
(\kappa(\theta)-\kappa^*)\|_2^2.
$$

The full checked proofs are
[the large-norm counterexample](../proofs/MATRIX_MQAR_LARGE_NORM_COUNTEREXAMPLE_PROOF.md)
and [the small-initialization counterexample](../proofs/MATRIX_MQAR_SMALL_INIT_COUNTEREXAMPLE_PROOF.md).
The independent 48-episode audit reports maximum discrepancy $1.11\times10^{-16}$.

## The remaining theorem

Write a nonzero initialization as $\theta_0=\varepsilon\xi$, with
$\xi\in\mathbb S^{41}$. The correct target is no longer an unconditional convergence
theorem. It is a basin classification: construct a checkable success set
$\mathcal G\subset\mathbb S^{41}$ and explicit failure sets $\mathcal F_j$ such that,
up to a null remainder,

$$
\xi\in\mathcal G
\Longrightarrow
\kappa(\phi_s(\varepsilon\xi))\to\kappa^*,
$$

whereas each $\xi\in\mathcal F_j$ has a proved wrong limiting kernel. The conditions
defining $\mathcal G$ must be derived from the data signal and factor dynamics; they
may not assume aligned attention, $K=Q$, a positive gain, or a uniform pullback bound.

The next proof obligation is precise: classify the sign/orientation of the contrast
flow near the score origin, prove retained contrast access on the candidate success
region, and show that all trajectories leaving that region enter one of the certified
failure basins. LEGO and depth composition remain blocked until this classification is
closed.

## What M1 adds

The M1 experiment uses public Zoology-compatible MQAR and a standard four-layer
Transformer. Across 20 paired seeds, exact $Q=K=0$ remains invariant and fails, while
a $2^{-8}$ nonzero Q/K scale escapes and learns. This validates the exact access
barrier in a less reduced architecture. It does not test the continuous-time basin
classification, and its AdamW trajectory is not evidence for gradient-flow convergence.
