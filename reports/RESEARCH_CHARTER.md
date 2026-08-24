# Research Charter: Training-Aware Transformer Dynamics

## Fixed conditional research question

> Under what explicit and checkable conditions does population gradient flow in a
> factorized exact-softmax Transformer learn a task-aligned $QK/OV$ interaction
> kernel, and when does that learned kernel implement the required interaction graph
> through network depth?

The object is fixed; the sufficient conditions are what the research must discover.
The minimal gates concern only the task, model class, and factorized optimization path.
Any additional constant must be derived from those objects or justified by an exact
impossibility result. Correct attention cannot appear among the assumptions.

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

## Structured task class

An input $X$ specifies a directed acyclic interaction graph whose nodes follow the
causal token order:

$$
G^*(X)=(V,E^*(X)),
\qquad
\max_v |\operatorname{pa}(v)|\le\Delta,
\qquad
\operatorname{depth}(G^*)\le D.
$$

Each correct node state is computed only from its declared parents and local input:

$$
z_v^*
=
\psi_v\left((z_u^*)_{u\in\operatorname{pa}(v)},\xi_v\right),
\qquad
\|z_v^*\|\le M.
$$

A declared edge must matter. Holding the other parent states and local input fixed,
resample the parent on that edge and require

$$
\kappa^2
=
\inf_{u\to v\in E^*}
\mathbb E\left[
\left\|
\psi_v\left((z_w)_{w\in\operatorname{pa}(v)},\xi_v\right)
-\psi_v\left((z_w^{u\leftarrow\widetilde u})_{w\in\operatorname{pa}(v)},\xi_v\right)
\right\|^2
\right]
>0.
$$

Here $\widetilde u$ is an independent draw from the conditional support of parent $u$
while the other parents and $\xi_v$ remain fixed. The data therefore determine a
necessary local interaction without assuming an attention mechanism.

The first two instances are deliberately simple and auditable:

| Data law | Graph parameters | Role |
|---|---|---|
| MQAR-compatible retrieval | $D=1$, $\Delta=1$ | discover one-step training conditions |
| cyclic LEGO state tracking | arbitrary $D$, $\Delta=2$ | test whether the conditions compose through depth |

For MQAR, let $N\ge m\ge2$ and sample

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

Fresh values prevent label memorization by key identity; random $J$ prevents a
fixed-position solution. The earlier stress-test value $N=32$ is not a theorem
assumption.


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

## Conditions to be established

The same task output can be implemented by several internal kernels. Before training,
register a bounded admissible state set $\mathcal Z_\ell$. For training time $s$, let
$\mathcal S_\ell(s)$ be the state distribution after the first $\ell$ learned layers,
with $\operatorname{supp}\mathcal S_\ell(s)\subseteq\mathcal Z_\ell$. Let
$m_i^{*,\ell}(Z)$ be the required message at receiver $i$ for $Z\in\mathcal Z_\ell$,
and define

$$
\mathfrak K_\ell^*
=
\left\{
\widetilde{\mathcal K}:
\sum_{j\le i}\widetilde{\mathcal K}(i,j;Z)z_j
=m_i^{*,\ell}(Z)
\ \text{for every }Z\in\mathcal Z_\ell
\right\}.
$$

The task-weighted kernel error is

$$
\eta_\ell(s)
=
\inf_{\widetilde{\mathcal K}\in\mathfrak K_\ell^*}
\left(
\mathbb E_{Z\sim\mathcal S_\ell(s)}
\left\|
\sum_{j\le i}
\left(\mathcal K_{\ell,s}-\widetilde{\mathcal K}\right)(i,j;Z)z_j
\right\|^2
\right)^{1/2}.
$$

This compares the message delivered to the task, not gauge-dependent raw factors.
The current three categories come from the actual task and model, not from a desired
proof shape:

| Candidate category | Concrete source | Present evidence | Current status |
|---|---|---|---|
| task identifiability | matched parent changes in MQAR and LEGO | $\kappa$ is computable from the exact population; deleting a zero-effect edge leaves the task unchanged | task-defined, cross-task form still to prove |
| finite representability | $\operatorname{rank}(B_h),\operatorname{rank}(C_h)\le d_h$ | known low-rank expressivity bounds and dense-versus-rank-matched controls | capacity term justified; general threshold unknown |
| factor access | exact $Q/K$ and $O/V$ composite dynamics | $Q=K=0$ barrier and the positive role-tied MQAR theorem | access is necessary; weakest sufficient condition unknown |

A condition enters the general theorem only after this chain is explicit: task/model
source, exact mathematical witness, experimental check where applicable, and prior-art
boundary. No empirical correlation is promoted directly to an assumption.

Let $\mathfrak K_\ell^{\rm model}$ be the kernels attainable under the chosen head
ranks. Define the a priori capacity error

$$
\varepsilon_{\rm cap}
=
\max_\ell
\inf_{\overline{\mathcal K}\in\mathfrak K_\ell^{\rm model}}
\inf_{\widetilde{\mathcal K}\in\mathfrak K_\ell^*}
\sup_{Z\in\mathcal Z_\ell}
\left\|
\sum_{j\le i}
\left(\overline{\mathcal K}-\widetilde{\mathcal K}\right)(i,j;Z)z_j
\right\|.
$$

This quantity is fixed before training. A positive or diverging score margin is a
conclusion to be derived when the task requires source selection; it is not an input
condition.

For factor access, the composite preconditioners are

$$
\mathcal P_B(G)=GK^{\top}K+Q^{\top}QG,
\qquad
\mathcal P_C(G)=GV^{\top}V+OO^{\top}G.
$$

For the projections $\Pi_B^*,\Pi_C^*$ onto task-required tangent modes, one
simple sufficient certificate on the registered parameter region is

$$
\langle G,\mathcal P_B(G)\rangle_F
\ge\mu_B\|\Pi_B^*G\|_F^2,
\qquad
\langle G,\mathcal P_C(G)\rangle_F
\ge\mu_C\|\Pi_C^*G\|_F^2.
$$

This certificate excludes the exact $Q=K=0$ barrier without requiring every raw
factor to be full rank. The barrier proves that some access condition is necessary; it
does not prove that uniform coercivity is the weakest possible condition.

A proof may introduce a vector $\varphi_*$ of source-versus-nuisance contrasts computed
from the chosen data law:

$$
\Sigma_*=\mathbb E[\varphi_*\varphi_*^{\top}].
$$

No covariance floor is assumed. If convergence needs one, it must be derived from the
task law or shown necessary by a counterexample before it enters a theorem statement.

To turn a kernel bound into a depth bound, write the target local update as
$\mathcal F_\ell^*(Z,m)$. The exact architecture must supply constants
$\Lambda_\ell,\Gamma_\ell$ satisfying

$$
\left\|\mathcal F_\ell^*(Z,m)-\mathcal F_\ell^*(Z',m')\right\|
\le
\Lambda_\ell\|Z-Z'\|
+\Gamma_\ell\|m-m'\|.
$$

This is a derived stability lemma, not an independent scientific assumption.

The symbol $\mathcal A$ denotes the fully specified Transformer architecture. The
target conditional theorem is

$$
\eta_\ell(s)
\le
r_\ell(s;\mathcal D,\mathcal A,\theta_0)
+\varepsilon_{\rm cap},
\qquad
\lim_{s\to\infty}r_\ell(s;\cdot)=0.
$$

The rate function must be derived from the data law and factorized gradient flow. It
may be polynomial rather than exponential; no rate is assumed in advance.

## Task alignment

For target slot $J$, define

$$
\gamma_s(X)=u_{qJ}(s;X)-\max_{j\ne J}u_{qj}(s;X).
$$

If the query sees $n_{\rm nt}$ non-target positions, exact softmax gives

$$
1-a_{qJ}\le n_{\rm nt}e^{-\gamma_s}.
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

The exact MQAR theorem is the first proved slice, not the final general theorem. It
uses a one-head role-tied parameterization with a learned radial dictionary scale. In
that slice the task graph is identifiable, capacity error is zero, and positive scalar
factors give access to the required score and value modes. The proved conclusion is

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

1. **Discover necessity on MQAR:** test which of identifiability, representability,
   and factor access cannot be removed.
2. **Lift to matrix factors:** derive margin and transport from those conditions for
   learned embedding directions and factorized $Q/K/O/V$ matrices.
3. **Compose through LEGO depth:** if the task-weighted layer error is $\eta_\ell(s)$
   and later target maps have Lipschitz constants $\Lambda_r$, prove

$$
\left(
\mathbb E\left\|\Phi_{\theta_s}^{L}(X)-\Phi^{*,L}(X)\right\|^2
\right)^{1/2}
\le
\sum_{\ell=0}^{L-1}
\Gamma_\ell
\eta_\ell(s)
\prod_{r=\ell+1}^{L-1}\Lambda_r.
$$

4. **Prove necessity:** construct exact counterexamples for degenerate initialization,
   insufficient rank, signed cancellation, bypasses, value-dependent scores, and
   nonidentifiable tasks.

The paper-level minimum is a condition theorem that covers a nontrivial family beyond
the role-tied MQAR slice and yields a training-to-depth corollary on LEGO. A clean
counterexample to one proposed condition is also progress; more model scale is not.

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
