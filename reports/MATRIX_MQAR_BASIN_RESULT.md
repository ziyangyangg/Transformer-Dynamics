# Matrix MQAR: Boundary Selection Result

## Verdict

The proposed theorem is false under “balanced, full-rank, nondegenerate
initialization.” Two exact obstructions are sufficient:

1. task-aligned softmax retrieval requires unbounded score margins, so the
   quotient coordinate $S$ cannot remain bounded;
2. Q/K Gram balance does not determine their task orientation. A balanced,
   full-rank, permutation-symmetric initialization with $K=-Q$ stays on a
   wrong invariant branch and cannot retrieve every ordered query correctly.

## 1. Correct retrieval forces $\lVert S\rVert_F\to\infty$

For an ordered target/distractor pair $(q,d)$, write

$$
a_{q\mid qd}=\frac{e^{S_{qq}}}{1+e^{S_{qq}}+e^{S_{qd}}},\qquad
b_{d\mid qd}=\frac{e^{S_{qd}}}{1+e^{S_{qq}}+e^{S_{qd}}}.
$$

The complete value cube gives the exact pair loss

$$
R_{qd}=\frac12\left[(g a_{q\mid qd}-1)^2+(g b_{d\mid qd})^2\right].
$$

Because the population is finite, $R\to0$ implies, for every $q\ne d$,

$$
g a_{q\mid qd}\to1,\qquad g b_{d\mid qd}\to0.
$$

Hence

$$
S_{qq}-S_{qd}
=\log\frac{a_{q\mid qd}}{b_{d\mid qd}}
=\log\frac{g a_{q\mid qd}}{g b_{d\mid qd}}
\longrightarrow+\infty.
$$

Therefore a compact LaSalle argument in $(S,g)$ cannot prove $R\to0$.
The correct object must either compactify the attention kernel or control an
unbounded margin trajectory.

## 2. Balanced full-rank counterexample

Let

$$
E(0)=I,\quad Q(0)=\alpha I,\quad K(0)=-\alpha I,\quad
O(0)=V(0)=I,\quad w(0)=\beta u,qquad \alpha,\beta>0.
$$

All factors are full rank and $QQ^\top=KK^\top$. The complete population and
this initialization are invariant under concept permutations. On this fixed
subspace the score gradient $G_B$ is symmetric. Since

$$
G_Q=K G_B^\top=-QG_B,\qquad G_K=QG_B,
$$

gradient flow satisfies $\dot K=-\dot Q$, so $K=-Q$ is preserved. Thus

$$
S=-E Q^\top Q E^\top\preceq0.
$$

For any two concepts $q\ne d$, the two required directed margins obey

$$
(S_{qq}-S_{qd})+(S_{dd}-S_{dq})
=(e_q-e_d)^\top S(e_q-e_d)\le0.
$$

They cannot both tend to $+\infty$. This is a balanced, full-rank wrong-boundary
counterexample, not a zero-factor or collapsed-dictionary pathology.

## 3. The canonical wrong boundary is a saddle, not an open attractor

Let $P_1=\mathbf 1\mathbf 1^\top/3$, $P_c=I-P_1$, and consider the exact
access-singular boundary

$$
E=e_1P_1+e_\perp P_c,\qquad Q=qP_1,\qquad K=-qP_1,
$$

with the value gain set to its uniform-attention least-squares optimum. Then

$$
R=\frac14,\qquad \nabla_\theta R=0,\qquad
G_S=-\frac18P_c\ne0.
$$

The quotient still requests a task-contrast update, but the Q/K contrast factors
have collapsed and cannot transmit it. For every matrix
$\Delta=P_c\Delta P_c$, the tied perturbation
$\delta Q=\delta K=\Delta$ satisfies

$$
\delta\dot Q=\frac{e_\perp^2}{8}\delta Q,\qquad
\delta\dot K=\frac{e_\perp^2}{8}\delta K.
$$

For $C=d=3$, this certifies $(C-1)^2=4$ independent unstable directions.
The full numerical Jacobian shows six positive directions, but only these four
are used as a theorem statement. Thus the counterexample invalidates an
all-initializations theorem, yet it does not exhibit an open bad basin.

## 4. What remains true on the positive symmetric branch

On the permutation-symmetric branch $K=Q$, let

$$
G_B=E^\top G_S E,\qquad \gamma=\partial_g R.
$$

The exact pullback identities give

$$
\lVert G_Q\rVert_F^2+\lVert G_K\rVert_F^2
\ge 2\sigma_{\min}(Q)^2\sigma_{\min}(E)^4\lVert G_S\rVert_F^2,
$$

and

$$
\lVert G_w\rVert^2+\lVert G_O\rVert_F^2+\lVert G_V\rVert_F^2
=c_g(\theta)\gamma^2,
$$

where

$$
c_g(\theta)=\lVert Cu\rVert^2
+\lVert w\rVert^2\lVert Vu\rVert^2
+\lVert O^\top w\rVert^2\lVert u\rVert^2.
$$

Thus the desired access inequality holds pointwise with

$$
\mu(\theta)=\min\left\{
2\sigma_{\min}(Q)^2\sigma_{\min}(E)^4, c_g(\theta)
\right\}.
$$

The missing step is uniform positivity along the trajectory. The invariant

$$
E^\top E-2Q^\top Q=\text{constant}
$$

does not by itself lower-bound both $E$ and $Q$: they may lose the same task
mode together.

## 5. Numerical audit and corrected theorem target

Two-tolerance float64 DOP853 agrees with the exact obstruction. At $s=64$:

| initialization | risk | target mass | distractor mass | tolerance discrepancy |
|---|---:|---:|---:|---:|
| $K=Q$ | $2.8111\times10^{-4}$ | 0.91081 | 0.02160 | $3.53\times10^{-11}$ |
| $K=-Q$ | 0.250022 | 0.32312 | 0.32314 | $2.84\times10^{-14}$ |

Both balance-invariant drifts are below $7.3\times10^{-16}$. These trajectories
verify the algebra; they are not the proof.

The positive symmetric branch is a proof scaffold, not a proposed assumption.
The broad target uses the initialization actually used in training: a continuous,
centered random law with matched factor scales, full score access, and nonzero
value access almost surely,

$$
\theta_0\sim\nu\ll\mathrm{Leb},\qquad
\Pr_\nu\!\left[\det(EQK)\ne0,\ c_g(\theta_0)>0\right]=1.
$$

Let $\mathcal Z_{\mathrm{wrong}}$ be all non-task-aligned stationary boundary
families in a compactified attention-kernel state space. The theorem target is

$$
\nu\!\left(W^{cs}(\mathcal Z_{\mathrm{wrong}})\right)=0
\quad\Longrightarrow\quad
\Pr_{\theta_0\sim\nu}\!\left[
R(s)\to0,\
S_{qq}(s)-S_{qd}(s)\to+\infty\ \forall q\ne d
\right]=1.
$$

What must be proved is that every wrong boundary family has an unstable normal
mode and that trajectories cannot escape through a new access-singular family.
The assumptions should remain task coverage, score/value role separation,
sufficient rank, and an absolutely continuous initialization—not $K=Q$, exact
permutation symmetry, or a postulated uniform access constant.
