# Method Specification

## 1. Target theorem

For $(X,Y)\sim\mathcal D$,

$$
R(\theta)=\frac12\mathbb E_{\mathcal D}
\left(f_\theta(X)-Y\right)^2,
\qquad
\dot\theta_s=-\nabla_\theta R(\theta_s).
$$

The primary objective is to prove, under explicit data, initialization, and scale
assumptions, that training creates a task-aligned interaction kernel and that the
resulting finite-depth network implements the known task graph $G^*(X)$.

The theorem must control

$$
\gamma_s(X)=u_{iJ^*}(s)-\max_{j\ne J^*}u_{ij}(s),
$$

$$
\mathcal E_{\rm transport}(s)
=
\left\|\mathcal K_s(i,J^*;X)z_{J^*}-m_i^*(X)\right\|^2
+
\sum_{j\ne J^*}
\left\|\mathcal K_s(i,j;X)z_j\right\|^2,
$$

and

$$
\mathcal E_{\rm depth}(s,L)
=
\left\|\Phi_{\theta_s}^{L}(X)-\Phi^*(X)\right\|.
$$

Risk reduction without structural bounds is insufficient.

## 2. MQAR-compatible population

Let $N\ge m\ge2$. An episode contains distinct keys and fresh binary values:

$$
c_1,\ldots,c_m\in[N],
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

The target slot changes across episodes, so position cannot solve the task. Fresh
values prevent concept identity from memorizing the label. This is a single-query,
binary-value specialization of public MQAR, not the complete Zoology sequence law.

## 3. Exact-softmax kernel

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
\mathcal K_{\ell}(i,j;X)
=
\sum_h a_{\ell h,ij}(X)C_{\ell h}.
$$

$B$ selects a source. $C$ transports its content. Raw factors are gauge-dependent, so
primary statements use $B$, $C$, $\mathcal K$, or function values.

If the loss depends on the factors only through $B$ and $C$, then

$$
\dot B=-G_BK^{\top}K-Q^{\top}QG_B,
\qquad
\dot C=-G_CV^{\top}V-OO^{\top}G_C,
$$

where $G_B=\partial R/\partial B$ and $G_C=\partial R/\partial C$. These identities
are exact but not closed until the gradients are expressed through task statistics.

## 4. Identifiability

Low risk and exact Walsh coefficients test functional dependence on the correct value.
Direct source dependence is measured by blocking each query-to-memory score edge,
renormalizing softmax, and recomputing every descendant:

$$
S_{\rm key}
=
\mathbb E\left[
Y(f-f^{(-J)})
-
\frac1{m-1}
\sum_{i\ne J}Y(f-f^{(-i)})
\right].
$$

Low risk implies positive direct-edge selectivity only under explicit value-path
identifiability, gain-sign, and no-bypass conditions. Signed cancellation gives an
exact zero-risk counterexample when these conditions are removed.

## 5. Registered MQAR theorem

In the permutation-symmetric one-head parameterization, the target score exceeds all
distractor and zero-value query-self scores by $\delta$. Define

$$
a(\delta)=\frac{e^\delta}{e^\delta+m},
\qquad
b(\delta)=\frac1{e^\delta+m},
$$

$$
f(v)=g\left(a v_J+b\sum_{i\ne J}v_i\right),
$$

$$
R_m(g,\delta)
=
\frac12\left[(ga-1)^2+(m-1)(gb)^2\right].
$$

Retain factorization and a learned radial dictionary scale:

$$
\delta=qk\rho^2,
\qquad
g=ovw.
$$

For positive factors and

$$
g_0
\left[
a(\delta_0)-\frac{m-1}{m}b(\delta_0)
\right]
<1,
$$

the required conclusion is

$$
\delta(s)\to+\infty,
\qquad
g(s)\to1,
\qquad
\mathcal E_{\mathcal K}(s)=2R_m(s)\to0.
$$

The unrestricted initialization claim is refuted by $q_0=k_0=0$: the composite score
gradient is nonzero while both score-factor gradients remain zero. The exact theorem
and proof are in reports/MQAR_KERNEL_LEARNING_THEOREM.md.

## 6. LEGO transition

Only after the MQAR theorem is established does the data law change to the published
LEGO state-tracking distribution. Variables are sampled without replacement, the
initial state is uniform, actions are sampled with replacement, and

$$
y_t=g_t(y_{t-1}).
$$

For each transition, the task graph has two required sources: predicate clause $t$ and
answer clause $t-1$. The next theorem must derive their kernel weights from training
and propagate the resulting per-step error through a chain of length $L$. Existing LEGO
learnability and length-generalization results are prior art.

## 7. Training and evidence rules

- Use exact population gradient flow whenever the support is feasible.
- Otherwise use counter-addressable fresh batches that approximate the same law.
- Record full state, source/configuration hashes, and a failure ledger.
- Treat independent training seeds as the only inferential units.
- Aggregate episodes, values, clauses, layers, heads, and checkpoints within seed.
- Use interventions for causal claims; attention mass is descriptive.
- Do not change a gate after inspecting its result.

## 8. Training-to-depth bridge

At fixed training time $s$, suppose layer $\ell$ approximates its target operator with
error $\eta_\ell(s)$ and later target layers have Lipschitz constants
$\Lambda_r$. The intended bound is

$$
\mathcal E_{\rm depth}(s,L)
\le
\sum_{\ell=0}^{L-1}
\eta_\ell(s)
\prod_{r=\ell+1}^{L-1}\Lambda_r.
$$

Every $\eta_\ell(s)$ must be derived from the training theorem. It cannot be assumed
by postulating a correct kernel.

## 9. Stop rules

No new model family, dataset, grid, or diagnostic is added unless it tests an explicit
quantity above. Pythia checkpoints are not seeds. Local hybrid patches are not additive
module attribution. Rare collisions are not promoted to a mechanism before replicated
high-precision evidence and a failed explanation by known kernel geometry.
