# Post-primary `I_swap` tail analysis specification

Status: **exploratory, nonblocking follow-up**. This analysis uses the raw `D` and
episode metadata saved by `phase2_swap_sensitivity.py`. It does not change the IID
K=64 primary population, any precision gate, the registered P8 estimand, or the
training-seed inference unit.

## 1. Conditional tail object

For query concept `q`, swapped distractor `c`, and newly inserted absent concept
`c'`, define the ordered triad and its conditional response

\[
T_{q,c\to c'}=(q,c,c'),\qquad
\mu_\theta(T)=\mathbb E[D_\theta(X)\mid T_{q,c\to c'}=T],
\]

where

\[
D_\theta(X)=\{f_\theta(X^{\mathrm{swap}})-f_\theta(X)\}^2.
\]

The persisted aggregation key further contains swapped slot `k`, its value `v_k`,
target slot `J`, and label `y`. We will report count, mean `D`, total contribution,
and fraction of total checkpoint `I_swap` for both the pure triad and the stratum
`(T,k,v_k,J,y)`. Sparse groups are descriptive only; no balanced resampling replaces
the IID primary law.

## 2. Tail concentration

For the `N` episode contributions at one checkpoint, report

\[
\mathrm{CV}(D)=\frac{s_D}{\bar D},\qquad
n_{\mathrm{eff}}=\frac{(\sum_iD_i)^2}{\sum_iD_i^2},
\]

the Gini coefficient, top-1 and top-10 episode shares, top-1% and top-10% shares,
and cumulative estimates at K=8/16/32/64 blocks. Rank the heaviest triads by
`sum(D)` and separately by conditional mean subject to a declared minimum count.

## 3. Geometry regressors

Let `e_a` denote the learned embedding of concept `a`,
`G=EE^T`, `B_{lh}=Q_{lh}^T K_{lh}`, and `C_{lh}=O_{lh}V_{lh}`. Candidate regressors
for each ordered triad are

\[
\begin{aligned}
r_G^{\rm old}&=G_{q,c}, & r_G^{\rm new}&=G_{q,c'},
&\Delta r_G&=G_{q,c'}-G_{q,c},\\
r_{lh}^{QK}&=z_q^TB_{lh}(z_{c'}-z_c),\\
r_{lh}^{OV}&=g_l^TC_{lh}(z_{c'}-z_c),
\end{aligned}
\]

where `z` is the exact normalized residual input to that layer and
`g_l=grad_{x_q^l} f_theta` is the local downstream readout gradient. We will also
include absolute scores, head maxima/sums, slot/value indicators, training step,
and interactions such as `Delta r_QK * r_OV`. Using `g_l` rather than the final
readout vector keeps the OV regressor valid through later nonlinear layers.

## 4. Statistical use and decision boundary

Fit weighted descriptive regressions to `log(D+epsilon)` and triad mean `D`, with
training-seed fixed effects and cluster/whole-seed uncertainty. Evaluate prediction
by held-out training seeds and held-out ordered triads; apply BH correction to any
declared coefficient family. Compare E-Gram-only, QK-only, OV-only, and joint models
out of sample.

These associations generate a mechanistic hypothesis only. They do not identify a
causal mediator: a later study must intervene on matched E/QK/OV geometry while
holding the retrieval function and on-manifold episode law fixed.
