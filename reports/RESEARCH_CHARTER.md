# Research Charter: Training-Aware Transformer Dynamics

## Fixed research question

> Can one derive, from a task distribution and population gradient flow, the
> factorized $QK/OV$ interaction kernel learned by an exact-softmax Transformer and
> prove that its layer dynamics implement the interaction graph required by the task?

This question is fixed. Retrieval, clustering, rank, heads, superposition, collisions,
and module patches are examples, assumptions, or diagnostics; none replaces the
question.

## Literature gap

*A Mathematical Perspective on Transformers* develops depth dynamics for prescribed
interaction laws. Section 10 of the current arXiv version identifies parameter training
as a major untreated problem; it does not state a numbered open problem asserting our
exact theorem.

Training theory already proves task-dependent attention selection in several special
settings: max-margin token selection, next-token co-occurrence, induction heads,
linear-regression ICL, multi-head allocation, and LEGO state tracking. Therefore
"attention can learn to select information" is not a contribution.

The unresolved joint map is

$$
\boxed{
(\mathcal D,R,\theta_0)
\xrightarrow{\text{training time }s}
\mathcal K_{\theta_s}
\xrightarrow{\text{network depth}}
\Phi_{\theta_s}^{L}(X)
}.
$$

The intended contribution is not a collection of future-work items. It must identify a
common mathematical object linking a training theorem to a fixed-kernel depth theorem.

## Variables and task

Let

$$
N\ge m\ge2,
\qquad
d,L,H,d_h\in\mathbb N,
\qquad
s\ge0,
\qquad
\ell\in\{0,\ldots,L-1\}.
$$

$N$ is key vocabulary size, $m$ is memory size, $d$ is residual width, $H$ is head
count, $d_h$ is per-head width, $s$ is training time, and $\ell$ is layer index.
The value $N=32$ used in an earlier stress test is not part of the theorem.

The MQAR-compatible population is

$$
c_1,\ldots,c_m\ \text{distinct in }[N],
\qquad
v_i\overset{\rm iid}{\sim}\operatorname{Unif}\{-1,+1\},
$$

$$
J\sim\operatorname{Unif}[m],
\qquad
q=c_J,
\qquad
Y=v_J.
$$

A memory token contains $(c_i,v_i)$; the query contains only $q$. Fresh values rule out
label memorization by key identity, and random $J$ rules out a fixed-position solution.
This is a single-query binary-value specialization of MQAR.

## Model and kernel

For $(X,Y)\sim\mathcal D$,

$$
R(\theta)
=
\frac12\mathbb E\left(f_\theta(X)-Y\right)^2,
\qquad
\dot\theta_s=-\nabla_\theta R(\theta_s).
$$

For layer $\ell$ and head $h$,

$$
B_{\ell h}=Q_{\ell h}^{\top}K_{\ell h},
\qquad
C_{\ell h}=O_{\ell h}V_{\ell h},
$$

$$
a_{\ell h,ij}
=
\frac{
\exp\left\{
\beta\nu(z_i^\ell)^{\top}B_{\ell h}\nu(z_j^\ell)/\sqrt{d_h}
\right\}
}{
\sum_{k\le i}
\exp\left\{
\beta\nu(z_i^\ell)^{\top}B_{\ell h}\nu(z_k^\ell)/\sqrt{d_h}
\right\}
},
$$

$$
\mathcal K_{\ell,s}(i,j;X)
=
\sum_h a_{\ell h,ij}(s;X)C_{\ell h}(s).
$$

$B$ determines which source is read; $C$ determines what is transported. The rank
bounds $\operatorname{rank}(B),\operatorname{rank}(C)\le d_h$ are parameterization
facts, not new results.

The exact composite identities are

$$
\dot B=-G_BK^{\top}K-Q^{\top}QG_B,
\qquad
\dot C=-G_CV^{\top}V-OO^{\top}G_C.
$$

They are only a starting point. The missing step is to express $G_B,G_C$ through a
small set of task statistics and prove the resulting trajectory.

## Task alignment

For target slot $J$, define

$$
\gamma_s(X)=u_{qJ}(s;X)-\max_{j\ne J}u_{qj}(s;X).
$$

If the query sees $M$ non-target positions, exact softmax gives

$$
1-a_{qJ}\le M e^{-\gamma_s}.
$$

This controls attention mass, not transported content. A value-channel transport error
is

$$
\mathcal E_{\mathcal K}(s)
=
\mathbb E\left[
\left\|P_{\rm val}\mathcal K_s(q,J)z_J-v_Jr_*\right\|^2
+
\sum_{i\ne J}
\left\|P_{\rm val}\mathcal K_s(q,i)z_i\right\|^2
\right].
$$

For a fixed key skeleton, Walsh coefficients satisfy

$$
2R_\omega
=
\left(\widehat f_{\{J\}}-1\right)^2
+
\sum_{S\ne\{J\}}\widehat f_S^2.
$$

This identifies functional dependence on values, not a unique internal route. The
registered direct-edge quantity is

$$
S_{\rm key}
=
\mathbb E\left[
Y(f-f^{(-J)})
-
\frac1{m-1}\sum_{i\ne J}Y(f-f^{(-i)})
\right],
$$

where each blocked model renormalizes softmax and recomputes every descendant.

## Current theorem status

The exact MQAR theorem is proved for a one-head role-tied parameterization with a learned radial
dictionary scale. For positive nondegenerate factors satisfying the registered
alignment condition,

$$
\delta(s)\to+\infty,
\qquad
g(s)\to1,
\qquad
\mathcal E_{\mathcal K}(s)=2R(s)\to0.
$$

Two exact obstructions delimit the result:

1. $Q=K=0$ is an invariant factorization barrier although the composite score gradient
   is nonzero.
2. Signed multi-head gains can yield $R=0$ but $S_{\rm key}=0$.

Thus arbitrary initialization and unqualified internal-route identification are false.
Arbitrary learned embedding directions, RMSNorm, residual bypasses, and multiple heads
remain open.

## Theorem program

1. **Training selection:** derive margin and transport from the task distribution.
2. **Factorization:** quantify the Gram preconditioners and distinguish conditioning
   from enlarged function class.
3. **Training-to-depth:** if layer error is $\eta_\ell(s)$ and later target maps have
   Lipschitz constants $\Lambda_r$, prove

$$
\mathcal E_{\rm depth}(s,L)
\le
\sum_{\ell=0}^{L-1}
\eta_\ell(s)
\prod_{r=\ell+1}^{L-1}\Lambda_r.
$$

4. **Necessity:** provide exact counterexamples for degenerate initialization, signed
   cancellation, bypasses, value-dependent scores, and nonidentifiable tasks.

The paper-level minimum is one theorem that closes training selection and
training-to-depth, with the necessary conditions stated explicitly.

## Data sequence

| Stage | Data | Role |
|---|---|---|
| A | MQAR-compatible exact population | one-step kernel-learning theorem |
| B | published LEGO state tracking | multi-step interaction graph and depth bridge |
| C | external structured tasks | only after the LEGO theorem |
| D | multi-seed pretrained trajectories | only after predictions are frozen |

LEGO has now entered only at the data-contract level: the exact cyclic-group population,
five-token clauses, and two source edges per transition are implemented. No LEGO
training claim has been made.

Synthetic data are appropriate because the probability law and target graph are
mathematical objects. Natural text usually lacks a uniquely auditable internal graph
and therefore cannot establish the first structural theorem.

## Position of existing evidence

| Evidence | Valid role | Invalid claim |
|---|---|---|
| fixed-kernel clustering | verifies prescribed-kernel depth dynamics | training learned that kernel |
| controlled retrieval | tests kernel quantities with known target edges | represents language generally |
| dense/rank-matched controls | tests a candidate function-class boundary | low rank itself is new |
| Walsh, swap, and edge blocking | separate functional use from direct paths | uniquely locate QK/OV/FFN |
| Pythia-70M checkpoints | calibrate measurements on a real model | one trajectory is a training law |

The remaining bottleneck is mathematical closure, not additional model scale.

## Stop rules

- Do not promote retrieval, rank, rare collisions, or a named compensator to the top
  question.
- Do not repackage fixed-QKV clustering, low-rank attention, or embedding
  nonorthogonality as contributions.
- Do not infer structural alignment from accuracy or attention maps.
- Do not add another model family or dataset before the LEGO theorem closes or fails
  with an exact counterexample.
