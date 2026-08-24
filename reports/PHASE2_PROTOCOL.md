# Phase-II Preregistered Controlled Study

Frozen 2026-08-20 before Phase-II production. Phase-II has a separate schema and does
not alter Phase-I artifacts.

## 1. Data, model, and risk

An episode is

$$
U=(c_{1:m},v_{1:m},J),
\qquad
c_i\ne c_j,
\qquad
v_i\in\{-1,+1\},
$$

$$
q=c_J,
\qquad
Y=v_J.
\tag{P1}
$$

The hard cell is

$$
(C,d,m,L,H)=(32,8,4,2,4),
\qquad
d_h=2.
\tag{P2}
$$

The population risk is always

$$
R(\theta)
=
\frac12\mathbb E_U
\left[f_\theta(U)-Y\right]^2.
\tag{P3}
$$

Any stored mean squared error must satisfy $R=\operatorname{MSE}/2$.

## 2. Functional routing

For a fixed key skeleton $(c_{1:m},J)$, enumerate all
$v\in\{-1,+1\}^m$ and define

$$
\widehat f_S(c,J)
=
2^{-m}
\sum_v
f(c,v,J)
\prod_{i\in S}v_i.
\tag{P4}
$$

Let

$$
E_T=\mathbb E(\widehat f_{\{J\}}-1)^2,
\qquad
L_D=\mathbb E\sum_{i\ne J}\widehat f_{\{i\}}^2,
$$

$$
L_H=\mathbb E\sum_{|S|\ge2}\widehat f_S^2,
\qquad
L_0=\mathbb E\widehat f_\varnothing^2,
\qquad
L_W=L_D+L_H+L_0.
\tag{P5}
$$

Parseval gives the exact audit

$$
2R=E_T+L_W.
\tag{P6}
$$

An independent target-value flip computes

$$
\Xi_{\rm value}
=
\frac12
\mathbb E
\left[
v_J\{f(c,v,J)-f(c,v^{\oplus J},J)\}
\right]
=
\mathbb E\widehat f_{\{J\}}.
\tag{P7}
$$

The two implementations must agree within $10^{-6}$.

A support-preserving distractor concept swap defines

$$
I_{\rm swap}
=
\mathbb E\left[f(U')-f(U)\right]^2.
\tag{P8}
$$

## 3. Direct key path

For every episode and memory slot $i$, set all final-query-to-slot-$i$ scores to
$-\infty$ in every layer and head, then recompute softmax and every descendant:

$$
\delta_i(U)
=
Y\left[
f(U)
-
f\left(
\operatorname{do}
\{s_{Ti}^{\ell h}=-\infty\ \forall\ell,h\}
\right)
\right].
\tag{P9}
$$

The registered estimand is

$$
S_{\rm key}
=
\mathbb E\left[
\delta_J
-
\frac1{m-1}\sum_{i\ne J}\delta_i
\right].
\tag{P10}
$$

Target effect, mean distractor effect, and their difference must all be stored.
Attention mass is descriptive and cannot replace (P10).

## 4. Random units and evaluation

Training seed is the independent unit. Episodes, value assignments, swap pairs, layers,
heads, and tokens are aggregated within seed.

Cohorts were frozen as discovery/remedy seeds 100-111, untouched confirmation seeds
1000-1023, and optimizer-replication seeds 2000-2015. Each master seed maps to distinct
initialization, training, evaluation, Walsh, swap, patch, and diagnostic streams.
Pairable arms use identical abstract episodes.

Each checkpoint uses at least 8192 base episodes, at least 512 key-target skeletons with
all 16 value assignments, and 2048 fixed swap pairs. Analysis uses float64. Required
numerical gates are

$$
\text{Parseval relative gap}<10^{-6},
\qquad
\text{finite-chord relative gap}<10^{-5}.
\tag{P11}
$$

## 5. Training limit and scheduler

Every new seed trains with AdamW, learning rate 0.003, and zero weight decay to step
800. The full model, optimizer, and data-stream state is then forked into a constant
schedule and preregistered cosine schedules. Checkpoints include

$$
s\in
\{0,25,50,100,200,400,800,1200,1600,2400,3200,4800,6400\}.
\tag{P12}
$$

For $Z\in\{R,L_W,I_{\rm swap}\}$, fit within seed

$$
\log_2\max\{Z_r(s),10^{-8}\}
=
a_{r,Z}
-
p_{r,Z}\log_2(s/800)
+
e
\tag{P13}
$$

at $s\in\{800,1600,3200,6400\}$. The rate differences are

$$
d_r^W=p_{r,L_W}-p_{r,R},
\qquad
d_r^{\rm swap}=p_{r,I_{\rm swap}}-p_{r,R}.
\tag{P14}
$$

Same-rate equivalence requires simultaneous 90% TOST intervals within
$[-0.25,0.25]$. A stable residual additionally requires 3200-to-6400 plateau
equivalence within a factor 1.25 and a simultaneous 95% final lower bound above

$$
\tau_W=\tau_{\rm swap}=2.5\times10^{-3}.
\tag{P15}
$$

## 6. Factorization versus function class

For every head,

$$
B_h=Q_h^{\top}K_h,
\qquad
C_h=O_hV_h,
\qquad
\operatorname{rank}(B_h),\operatorname{rank}(C_h)\le d_h.
\tag{P16}
$$

Three arms were registered:

1. factorized Q/K and O/V;
2. dense direct $B_h,C_h\in\mathbb R^{d\times d}$, an upper bound that changes
   both conditioning and capacity;
3. rank-matched direct composites, retracted by truncated SVD to rank at most $d_h$.

All arms must start from identical composites and predictions:

$$
B_h(0)=Q_h(0)^{\top}K_h(0),
\qquad
C_h(0)=O_h(0)V_h(0),
\tag{P17}
$$

with maximum step-zero prediction gap below $10^{-6}$.

For $Z\in\{L_W,I_{\rm swap}\}$,

$$
\Delta_Z^{\rm rank}
=
\mathbb E_r
\log_2
\frac{
Z_r^{\rm rank\ direct}(6400)
}{
Z_r^{\rm factorized}(6400)
}.
\tag{P18}
$$

A conditioning remedy requires function noninferiority, at least a twofold residual
reduction, a simultaneous 95% interval in the improving direction, and at least 80%
of seeds passing the function gate. If only dense direct composites repair the
residual, the classification is rank/function capacity, not factorization conditioning.

## 7. Representation and head controls

For $C=32,d=8$, the Welch lower bound is

$$
\mu_W
=
\sqrt{\frac{C-d}{d(C-1)}}
\approx0.3111.
\tag{P19}
$$

A registered low-coherence frame must satisfy

$$
\mu(E)\le1.25\mu_W,
\qquad
\frac{
\left\|E^{\top}E-(C/d)I\right\|_F
}{
\left\|(C/d)I\right\|_F
}
\le0.02.
\tag{P20}
$$

The true orthogonal control is restricted to $C=d=8$ and cannot be interpreted as a
single-factor comparison with $C=32$.

For attention inner width $p=Hd_h$, bias-free attention has

$$
P_{\rm att}=4dp.
\tag{P21}
$$

Head controls separate fixed residual width, fixed per-head width, and fixed total
attention-plus-FFN parameter budget. A fixed-budget comparison satisfies

$$
4dp+2dr=2d(2p+r),
\qquad
2p+r=40.
\tag{P22}
$$

These are capacity-allocation controls, not a pure causal effect of head count.

## 8. Finite local diagnostics

For an on-support swap, the exact asymmetric attention decomposition is

$$
\delta m_C=\sum_i a_i(z_i'-z_i),
$$

$$
\delta m_R=\sum_i(a_i'-a_i)z_i,
$$

$$
\delta m_I=\sum_i(a_i'-a_i)(z_i'-z_i),
$$

$$
m'-m=\delta m_C+\delta m_R+\delta m_I.
\tag{P23}
$$

The QK contrast uses actual nonlinear suffix reruns from a common base state:

$$
C_{QK}^{\rm finite}
=
\mathbb E
\log
\frac{
p_{C+I}^2+10^{-12}
}{
p_{C+R+I}^2+10^{-12}
}.
\tag{P24}
$$

The OV directional statistic is

$$
g_{\rm swap}
=
\frac{\|C_h\delta m_h\|^2}
{\|\delta m_h\|^2+10^{-12}},
\qquad
g_{\rm iso}=\frac{\|C_h\|_F^2}{d},
$$

$$
A_{OV}
=
\mathbb E
\log
\frac{g_{\rm iso}+10^{-12}}
{g_{\rm swap}+10^{-12}}.
\tag{P25}
$$

For an FFN residual branch,

$$
\delta x_{\rm skip}=x'-x,
$$

$$
\delta x_{\rm ffn}
=
L^{-1/2}
\left[
F(N(x'))-F(N(x))
\right].
\tag{P26}
$$

The actual suffix responses $p_{\rm skip},p_{\rm ffn},p_{\rm joint}$ must report

$$
p_{\rm nonlin}
=
p_{\rm joint}
-
p_{\rm skip}
-
p_{\rm ffn}.
\tag{P27}
$$

These QK, OV, and FFN quantities are overlapping local hybrid estimands, not an
additive module attribution. A compensator claim additionally requires upstream energy,
tangent/finite agreement, pairwise direction agreement, simultaneous correction,
practical attenuation, functional validity, and replication.

## 9. Exact-population gradient-flow bridge

For small $C,m$, the complete support has size

$$
|\Omega|
=
\frac{C!}{(C-m)!}m2^m.
\tag{P28}
$$

Registered populations were $(C,m)=(4,2)$ and $(6,3)$. Explicit Euler reference
trajectories use

$$
\theta_{k+1}
=
\theta_k-\eta\nabla R(\theta_k),
\tag{P29}
$$

with Hessian-calibrated initial step size and step-halving comparisons. The registered
order-parameter vector is

$$
z=
(
R,K_{\rm target},L_D,L_H,\Xi_{\rm value},S_{\rm key},
r_{\rm eff}(E),\|B\|_F,\|C\|_F,
S_Q-S_K,S_O-S_V
).
\tag{P30}
$$

Every coordinate must pass the step-halving discrepancy threshold 0.10 before an
empirical closure field is fitted. Closure fitting uses discovery-only standardization
and untouched-only evaluation. Raw-coordinate error is a mandatory nongating
sensitivity. If numerical convergence fails, no closure theorem or closure failure may
be claimed.

## 10. Multiplicity and function gate

The function gate is

$$
A\ge0.95,
\qquad
R\le0.01,
\qquad
\Xi_{\rm value}\ge0.90.
\tag{P31}
$$

Primary paired inference uses whole-seed bootstrap resampling and simultaneous max-T
correction. Episodes and checkpoints never increase the inferential sample size.
Unregistered spectra or head patterns remain exploratory.

## 11. Open-problem escalation rule

A residual could be promoted to an empirical open problem only if all preregistered
checks passed: function validity, a positive final residual floor, late plateau,
failure of constant and cosine schedules, failure of rank-matched direct composites,
failure of representation and head-capacity controls, replication across optimizers
and FFN settings, true slot-wise $S_{\rm key}$, valid finite suffixes, complete
precision/replay audits, and a renewed prior-art search.

The resulting classifications were fixed in advance:

- longer training or cosine repair: incomplete convergence;
- rank-matched direct repair: factorization conditioning;
- dense-only repair: rank/function capacity;
- low-coherence repair: dictionary geometry;
- fixed-per-head-width repair: per-head bottleneck;
- no unique local module: distributed or nonidentifiable compensation.

## 12. Completed-study interpretation

The registered data do not support a stable irreducible residual. Cosine scheduling and
continued training improve exact-cube risk and Walsh leakage. Dense direct composites
strongly reduce residuals, whereas rank-matched direct composites do not. Therefore
the controlled evidence supports a rank/function-class boundary as a candidate
assumption; it does not support factorization conditioning as the sole explanation.

High-sample swap evaluation preserves the dense-versus-rank-matched direction but
remains heavy-tailed at a small number of checkpoints. Swap-specific population claims
are therefore descriptive. No QK, OV, or FFN compensator is identified.
