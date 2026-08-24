# Matrix MQAR at $C=3,m=2$: Exact Gradients and Obstructions

## Verdict

The unrestricted matrix kernel-learning claim is false, even on the smallest
complete MQAR population.  The identifiable function quotient has no finite critical
point, but the factorized parameterization has exact non-task-aligned critical
manifolds, and the quotient has a wrong asymptotic boundary where the gradient
vanishes.  Therefore a global LaSalle argument requires both factor-access and
correct-boundary conditions.

## 1. Frozen object

The 48 episodes and the role-separated exact-softmax model are defined in
[MATRIX_MQAR_C3M2_SPEC.md](MATRIX_MQAR_C3M2_SPEC.md).  The only identifiable
coordinates are

$$
S=EBE^\top,
\qquad
g=w^\top C u,
\qquad
B=Q^\top K,
\qquad
C=OV.
$$

For query concept $q$, target memory $q$, distractor $d\ne q$, and fixed zero-score
query-self mass,

$$
a=\frac{e^{S_{qq}}}{1+e^{S_{qq}}+e^{S_{qd}}},
\qquad
b=\frac{e^{S_{qd}}}{1+e^{S_{qq}}+e^{S_{qd}}}.
$$

After exact averaging over the two Rademacher values,

$$
R(S,g)
=
\frac1{12}\sum_{q\ne d}
\left[(ga-1)^2+(gb)^2\right],
\qquad
\mathcal E_{\mathcal K}=2R.
\tag{1}
$$

## 2. Exact population gradients

For one episode define

$$
r=\sum_i a_i v_i,
\qquad
\epsilon=gr-Y,
\qquad
\lambda_i=\epsilon g a_i(v_i-r),
\qquad
\gamma=\mathbb E[\epsilon r].
$$

Direct differentiation gives

$$
G_B=\mathbb E\sum_i\lambda_i e_qe_{c_i}^\top,
\tag{2}
$$

$$
G_{e_c}
=
\mathbb E\left[
\mathbf1\{q=c\}\sum_i\lambda_iBe_{c_i}
+
\sum_i\mathbf1\{c_i=c\}\lambda_iB^\top e_q
\right],
\tag{3}
$$

$$
G_C=\gamma wu^\top,
\qquad
G_w=\gamma Cu.
\tag{4}
$$

These formulas agree with float64 automatic differentiation on all entries.  Their
factor pullback is

$$
G_Q=KG_B^\top,
\quad
G_K=QG_B,
\quad
G_O=G_CV^\top,
\quad
G_V=O^\top G_C.
\tag{5}
$$

Consequently,

$$
\dot B=-G_BK^\top K-Q^\top QG_B,
\qquad
\dot C=-G_CV^\top V-OO^\top G_C.
\tag{6}
$$

## 3. Critical-point classification

### Proposition 1: no finite quotient critical point

Let $c=1-a-b>0$ be the query-self softmax mass.  If $g=0$, then

$$
\partial_gR=-\mathbb E[a]<0.
$$

If $g\ne0$ and an off-diagonal derivative vanishes, direct substitution into the
corresponding target-score derivative gives

$$
\partial_{S_{qd}}R_{qd}=0
\quad\Longrightarrow\quad
\partial_{S_{qq}}R_{qd}=g^2bc>0.
\tag{7}
$$

Every $S_{qd}$ with $q\ne d$ is a separate coordinate, whereas $S_{qq}$ sums two
strictly positive terms.  Hence $\nabla_{S,g}R\ne0$ at every finite $(S,g)$.

### Proposition 2: exact non-aligned parameter critical families

Let $\Psi(E,Q,K,O,V,w)=(S,g)$.  Although
$\nabla R(\Psi)\ne0$, the pullback

$$
\nabla_\theta R=J_\Psi(\theta)^\top\nabla_{S,g}R
\tag{8}
$$

can vanish.  Four exact families occur:

| Family | Exact condition | Risk | Lost access |
|---|---|---:|---|
| collapsed dictionary | $e_0=e_1=e_2=e$, $g=(2a)^{-1}$ | $1/4$ | concept contrasts |
| zero QK factors | $Q=K=0$, $g=3/2$ | $1/4$ | score-composite gradient |
| dead value path | $w=0$ and $Cu=0$ | $1/2$ | gain gradient |
| zero OV factors | $O=V=0$ | $1/2$ | value-composite gradient |

For each representative, every parameter gradient is zero while the quotient
gradient is nonzero.  The structural classification is therefore complete:

$$
\text{finite parameter critical point}
\Longrightarrow
\text{task aligned or access singular}.
\tag{9}
$$

The table gives exact irreducible witnesses; it is not claimed to be a full algebraic
decomposition of every access-singular factor tuple.

## 4. Why LaSalle is not yet sufficient

There is also a wrong noncompact boundary.  Set

$$
S=-t\mathbf1\mathbf1^\top,
\qquad
g=0.
$$

Then all memory attention vanishes into query-self and

$$
R\longrightarrow\frac12,
\qquad
\|\nabla_{S,g}R\|\longrightarrow0
\quad(t\to\infty).
\tag{10}
$$

Thus “no finite critical point” does not imply correct kernel learning.  A positive
matrix theorem must establish two separate properties along the actual trajectory:

$$
\left\|J_\Psi^\top\nabla_{S,g}R\right\|^2
\ge
\mu\left\|\nabla_{S,g}R\right\|^2
\quad\text{away from }\mathfrak K^*,
\tag{11}
$$

and a forward-invariant condition that excludes the self-only boundary (10).  The
first is factor access; the second is correct-boundary selection.  Neither may be
replaced by nonzero parameter norms alone.

## 5. Adaptive-ODE audit

One full-rank nondegenerate initialization was integrated on $s\in[0,10]$ by DOP853
with $(\mathrm{rtol},\mathrm{atol})=(10^{-9},10^{-11})$ and independently with
$(10^{-11},10^{-13})$.

| Quantity | $s=0$ | $s=10$ |
|---|---:|---:|
| $R$ | $0.2795433344$ | $0.0064576636$ |
| $\mathcal E_{\mathcal K}$ | $0.5590866687$ | $0.0129153272$ |
| target attention | $0.4029599112$ | $0.7544373258$ |
| distractor attention | $0.2985200444$ | $0.0861151860$ |

The maximum two-solve relative discrepancy was $7.54\times10^{-16}$ and the maximum
QK/OV balance-invariant drift was $4.34\times10^{-16}$.  The audit passed.  This
trajectory verifies the equations and numerical implementation only; it does not
override Propositions 1--2 or prove convergence.

## Resulting theorem target

The next positive statement must be basin-conditional:

> Derive from the MQAR data law and initialization a forward-invariant region in
> which factor access is bounded below and the self-only boundary is unreachable;
> then prove that the only remaining asymptotic kernel is task aligned.

Without both clauses, the proposed general matrix kernel-learning theorem is false.
