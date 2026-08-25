# Main Theorem Roadmap: From Task Law to Learned Interaction Kernel

> **Superseded target.** The all-regular-initialization and small-initialization
> density conjectures described below were refuted by verified open-basin
> counterexamples. The active theorem is the initialization-direction basin
> classification in [SINGLE_LAYER_THEORY_STATUS.md](SINGLE_LAYER_THEORY_STATUS.md).
> The dependency analysis below is retained as a record, not as a valid convergence
> claim.

Last primary-source audit: 2026-08-25.

## 1. The fixed problem

The project asks one question:

> Given a structured task distribution, when does population gradient flow of a
> factorized exact-softmax Transformer learn the task-required interaction kernel?

The first theorem is deliberately minimal. Let

$$
C=d=3,\qquad m=2,\qquad |\Omega|=(3)_2\,2\,2^2=48.
$$

For each episode, two distinct concepts $c_1,c_2$ carry independent values
$v_1,v_2\in\{-1,+1\}$. A uniformly sampled target $J\in\{1,2\}$ gives query
$q=c_J$ and label $Y=v_J$. The one-head model is

$$
S=EQ^\top K E^\top,\qquad g=w^\top OVu,
$$

$$
a_i=
\frac{\exp S_{q c_i}}
{1+\exp S_{q c_1}+\exp S_{q c_2}},qquad
f_\theta=g\sum_{i=1}^2 a_i v_i,
$$

$$
R(\theta)=\frac12\mathbb E_\Omega(f_\theta-Y)^2,qquad
\dot\theta=-\nabla R(\theta).
$$

Because the complete value cube is used,

$$
2R
=
\mathbb E_\Omega\sum_{i=1}^2
\left(ga_i-\mathbf 1\{i=J\}\right)^2.
$$

Thus zero risk is exactly correct value retrieval, not merely correct classification.
The intended first theorem is an almost-everywhere statement on the finite
regular-access set; equivalently, it applies to every absolutely continuous
initialization law $\nu$ supported on that set:

$$
\Pr_{\theta_0\sim\nu}\left[
\max_{\omega\in\Omega}\max_{i\in\{1,2\}}
\left|g(s)a_i(s;\omega)-\mathbf 1\{i=J\}\right|\to0
\right]=1.
\tag{T}
$$

Consequently, $R(\theta_s)\to0$ and
$S_{qq}(s)-S_{qd}(s)\to+\infty$ for every $q\ne d$. The task does not identify
$g$ separately: correct delivered coefficients are compatible with $g\to\infty$.
Selecting $g=1$ is therefore a secondary implicit-bias question, not part of (T).

The theorem must derive its basin from the data law, factor dynamics, and
initialization. It may not assume aligned attention, $K=Q$, permutation symmetry, or
a uniformly positive pullback constant.

## 2. What is already established

The complete population, gauge group, target kernel, exact factor gradients, and
float64 adaptive-ODE audit are frozen. Three facts now determine the proof strategy.

First, correct retrieval is noncompact:

$$
S_{qq}-S_{qd}=\log\frac{a_{q\mid qd}}{b_{d\mid qd}}\to+\infty.
$$

Therefore boundedness of $(S,g)$ is false and compact LaSalle cannot prove (T).

Second, balanced full-rank initialization is insufficient. The invariant branch

$$
E=I,qquad Q=\alpha I,qquad K=-\alpha I
$$

has $S=-EQ^\top QE^\top\preceq0$ and cannot make both directed margins of any
concept pair positive.

Third, the canonical uniform wrong boundary has

$$
R=\frac14,qquad \nabla_\theta R=0,qquad G_S=-\frac18P_c\ne0,
$$

but possesses at least four analytically certified unstable normal modes. Hence it
refutes an all-initializations theorem without yet refuting the almost-everywhere
target (T).

## 3. Dependency graph

The proof must proceed in the following order:

$$
\text{exact quotient}
\longrightarrow
\text{compactified boundary}
\longrightarrow
\text{all wrong strata}
\longrightarrow
\text{local stability}
\longrightarrow
\text{global exclusion}
\longrightarrow
\text{a.e. kernel learning}.
$$

Only after this chain closes may the project generalize $C,m,d$, pass from
population flow to SGD, or enter LEGO depth composition.

## 4. Work packages

### P0. Exact functional reduction — complete

**Question.** Which parameter changes are functionally observable?

**Tasks.** Derive the complete-population risk; verify $G_E,G_B,G_C,G_w$; quotient
the $Q/K$, $O/V$, embedding-coordinate, and gain gauges; prove $2R$ equals target
kernel error.

**Acceptance.** Symbolic formulas agree with enumeration and autograd in float64;
all registered gauge transformations preserve predictions and risk.

**Literature.** [Zoology](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html)
defines MQAR as a structured recall task. It does not analyze factorized Transformer
gradient flow. [Deep matrix factorization](https://papers.neurips.cc/paper/8960-implicit-regularization-in-deep-matrix-factorization)
motivates quotient observables but studies linear sensing/completion, not softmax
retrieval.

### P1. Compactify the correct and wrong boundaries — next

**Question.** What replaces bounded $(S,g)$ when correct score margins diverge?

**Small problems.**

1. Express risk on the attention simplex and identify every zero-risk face.
2. Separate margin magnitude $\rho$ from margin direction $\widehat M$.
3. Derive the exact factor flow in $(a,\widehat M,\rho,g)$, with a time rescaling if
   necessary.
4. Prove that every finite-risk trajectory has a nonempty limit set in the
   compactified state.

**Acceptance.** The transformed vector field extends continuously to every boundary
face used later; zero risk corresponds to one explicitly identified correct face.

**Literature.** Linear separable models exhibit diverging parameters but convergent
directions: [Soudry et al.](https://arxiv.org/abs/1710.10345) and
[directional convergence in homogeneous networks](https://proceedings.neurips.cc/paper_files/paper/2020/hash/c76e4b2fa54f8506719a5c0dc14c2eb9-Abstract.html).
[Max-Margin Token Selection](https://proceedings.neurips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html)
applies this idea to simplified attention. None covers jointly learned $E,Q/K,O/V$
under squared retrieval risk, so it supplies a method rather than the theorem.

### P2. Classify the factor-access singular set

**Question.** At which parameter boundaries can the quotient request a correction
while the factors cannot realize it?

**Small problems.**

1. Decompose concept space into the trivial and contrast representations of $S_3$.
2. Stratify by ranks and relative orientations of $E,Q,K$ and by value access
   $c_g$.
3. Solve $\nabla_\theta R=0$ on each stratum, modulo gauge and concept permutation.
4. Certify exhaustiveness with symbolic elimination and interval-checked witnesses.

**Acceptance.** A finite list of stationary families covers every real critical
point of the $C=3,m=2$ model. Each family records risk, quotient gradient, access
failure, dimension, and isotropy group.

**Literature.** Equivariant bifurcation and isotypic decompositions are standard in
[Golubitsky, Stewart, and Schaeffer](https://link.springer.com/book/10.1007/978-1-4612-4574-2).
[Symmetry-induced critical manifolds](https://proceedings.mlr.press/v139/simsek21a.html)
show why parameter symmetries create non-isolated neural-network critical sets.
[Algorithms in Real Algebraic Geometry](https://doi.org/10.1007/3-540-33099-2)
provides exact component and quantifier-elimination tools. These sources do not
classify the softmax MQAR singular set.

### P3. Determine the local stability of every wrong stratum

**Question.** Does any wrong family attract a positive-measure set?

**Small problems.**

1. Compute the normal linearization after removing gauge-tangent directions.
2. Produce one explicit unstable normal mode for every wrong stratum, or exhibit a
   genuine attracting counterexample.
3. Treat zero eigenvalues with center-manifold or higher-order analysis.
4. Bound the dimension of each center-stable set.

**Acceptance.** Every wrong family is either proved measure-zero attracting under
$\nu$, or it supplies an exact counterexample to (T).

**Literature.** The stable-manifold argument of
[Lee et al.](https://proceedings.mlr.press/v49/lee16.html) handles strict saddles;
[Panageas and Piliouras](https://arxiv.org/abs/1605.00405) extends measure-zero
avoidance to non-isolated critical sets and invariant regions. These results become
applicable only after P2 supplies an exhaustive list and P3 proves normal
instability; they do not themselves exclude boundary escape.

### P4. Prove global noncollapse and exclude wrong escape

**Question.** Can a generic trajectory lose a task contrast or value direction before
reaching the correct boundary?

**Small problems.**

1. Use exact Gram invariants to control gauge growth without claiming parameter
   boundedness.
2. Derive a task-direction lower bound on integrated factor access, weaker than a
   postulated uniform $\mu>0$.
3. Prove global existence and precompactness of normalized factors.
4. Show that every $\omega$-limit lies in the correct face or a P2 wrong stratum.
5. Exclude heteroclinic escape through successively lower-rank access strata.

**Acceptance.** Outside the P3 center-stable sets, no trajectory reaches or
asymptotically shadows a new access-singular boundary.

**Literature.** [Automatic balancing in homogeneous models](https://papers.nips.cc/paper_files/paper/2018/hash/fe131d7f5a6b38b23cc967316c13dae2-Abstract.html)
proves Gram/norm invariants; [implicit regularization in deep matrix
factorization](https://arxiv.org/abs/1905.13655) studies product dynamics.
[Łojasiewicz convergence](https://dial.uclouvain.be/pr/boreal/en/object/boreal%3A38704)
gives point convergence for bounded analytic gradient flows; the boundedness premise
is exactly what fails for the score margin here. The required
projective noncollapse argument is therefore new work, not an automatic corollary.

### P5. Close the minimal almost-everywhere theorem

**Question.** Do P1--P4 imply (T)?

**Small problems.** Prove monotone risk dissipation; combine compactified limit-set
classification with measure-zero wrong basins; prove uniform convergence of all
twelve delivered target/distractor coefficients to the task kernel; derive all six
directed margin limits. Analyze selection of $g$ only as a separate implicit-bias
corollary.

**Acceptance.** A proof for $C=3,m=2,d=3$ under a precisely stated natural random
initialization, plus an exact counterexample showing why each hypothesis is needed.
Adaptive ODE trajectories are used only to falsify intermediate lemmas.

**Literature boundary.** Existing Transformer training theorems already prove token
selection or task structure in special models: co-occurrence
[gradient flow](https://proceedings.neurips.cc/paper_files/paper/2024/hash/520416e27d3b0cef3cd70a083e2991c7-Abstract-Conference.html),
[learned embeddings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/29be2340edb8b224728248d4bb1ea9d4-Abstract-Conference.html),
[induction heads](https://arxiv.org/abs/2409.10559), and multi-head linear ICL
[training dynamics](https://openreview.net/forum?id=3TM3fxwTps). The increment in P5
is the exact boundary selection of a learned dictionary and factorized $QK/OV$ on an
identifiable retrieval population.

### P6. Generalize the MQAR theorem without changing the task family

**Question.** Which parts of P5 survive for general $C,m,d$ and finite head rank?

**Small problems.**

1. Determine the minimum composite rank needed for all directed MQAR margins.
2. Compute the positive capacity error below that rank.
3. Replace the $S_3$ decomposition by irreducible $S_C$ modes.
4. Identify which singular strata persist for arbitrary $m$.
5. Add multiple heads only if one head cannot represent the target kernel.

**Acceptance.** A task-dependent threshold separates impossible, representable but
inaccessible, and almost-everywhere learnable regimes.

**Literature.** The [low-rank attention bottleneck](https://proceedings.mlr.press/v119/bhojanapalli20a.html)
establishes an architectural expressivity limitation. It does not determine the
MQAR target rank or whether factorized training reaches a realizable target. Rank is
therefore a known constraint, not the claimed contribution.

### P7. Bridge population flow to sampled training

**Question.** Does finite-sample SGD track the population basin rather than create a
new conclusion?

**Small problems.** Freeze a finite horizon before margins diverge; prove uniform
gradient concentration on the normalized invariant region; couple SGD to the ODE;
then prove that stochastic noise does not stabilize a P3 wrong saddle.

**Acceptance.** A sample/step-size bound preserves the P5 basin classification with
high probability. Episodes and checkpoints are never treated as independent seeds.

**Literature.** [Benaïm's dynamical-systems treatment of stochastic
approximation](https://numdam.org/item/SPS_1999__33__1_0/) relates stochastic
iterates to mean ODEs. [Pemantle](https://www2.math.upenn.edu/~pemantle/papers/nonconvergence.pdf)
proves nonconvergence to linearly unstable points under sufficiently rich noise.
Their assumptions must be verified for MQAR minibatches rather than quoted.

### P8. Enter LEGO only after P5

**Question.** If one learned kernel performs one correct local retrieval/update, how
does its error compose through depth?

**Small problems.** Derive the two-parent LEGO local kernel; prove its one-step
factorized learning theorem using the P5 conditions; bound the propagation of routing
and local-operator errors over depth $L$.

**Acceptance.** For explicit constants derived from the published LEGO law,

$$
\|\widehat z_L-z_L^*\|
\le
\sum_{\ell=1}^L
\left(\prod_{r=\ell+1}^L L_r\right)
(\eta_\ell+\varepsilon_{\mathrm{op},\ell}).
$$

**Literature.** [Transformers Provably Learn Chain-of-Thought Reasoning with Length
Generalization](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b86a195e70f27017c514fa0e5f80595f-Abstract-Conference.html)
already proves attention concentration and length generalization for LEGO.
[A Mathematical Perspective on Transformers](https://arxiv.org/html/2312.10794v5)
and the [Transformer PDE](https://arxiv.org/abs/2501.18322) analyze depth dynamics
with prescribed interactions. Our only admissible increment is a single theorem
linking a learned factorized kernel to the finite-depth error; merely repeating LEGO
learnability is not new.

## 5. Immediate sequence and stop rules

The next four deliverables are P1.1 (attention-simplex boundary), P2.1 ($S_3$ mode
decomposition), P2.2 (rank/access strata), and P3.1 (normal Hessian of the known
uniform wrong family). P1 and P2 may proceed in parallel on paper, but P3 claims wait
for P2 exhaustiveness and P4 waits for the compactification in P1.

Stop and extract a missing condition if any of the following occurs:

1. a wrong stratum has a positive-measure stable basin;
2. normalized factors can escape without approaching a classified boundary;
3. finite representability fails at the registered rank;
4. the proof requires assuming the desired attention pattern.

Do not add a model, dataset, head, layer, FFN, or normalization to repair a failed
lemma. First state the exact counterexample in the frozen model.

## 6. Research map

| Category | Current content |
|---|---|
| Solved | exact MQAR population/quotient; reduced positive-branch theorem; bounded-quotient impossibility; balanced full-rank wrong branch; one wrong saddle's four unstable modes |
| Existing theory can be adapted | implicit margin direction; factor balancing; stable-manifold avoidance; equivariant mode decomposition; stochastic approximation |
| Known experimentally, not a theorem | dense composites and $H=1$ reduce toy leakage; high-$N$ swap effects are heavy-tailed; Pythia routing is nonmonotone on one trajectory |
| Main open theorem | almost-everywhere boundary selection for factorized matrix MQAR, followed by a learned-kernel-to-depth LEGO bound |

Pythia, scaling, and localization results are boundary evidence only. They do not
decide P1--P5 and are not prerequisites for the theorem.

The exact lemma contracts, counterexample exits, and dependency graph are frozen in
[`MATRIX_MQAR_PROOF_DECOMPOSITION.md`](MATRIX_MQAR_PROOF_DECOMPOSITION.md).
