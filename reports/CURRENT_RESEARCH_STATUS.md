# Current Theory--Experiment Status

**Frozen on 2026-08-26.** This document is the shortest authoritative account of
what the project has established, what the experiments mean, and what remains open.
It does not replace the detailed proofs, protocols, or result artifacts.

## 1. The fixed research question

The project studies the training side of Transformer interaction dynamics:

$$
(\mathcal D,R,\theta_0)
\xrightarrow{\text{gradient-flow time }s}
\{B_{\ell h}(s),C_{\ell h}(s)\}
\xrightarrow{\text{softmax}}
\mathcal K_{\ell,s}
\xrightarrow{\text{depth }\ell}
\Phi_{\theta_s}^{L}(X),
$$

$$
B_{\ell h}=Q_{\ell h}^{\top}K_{\ell h},
\qquad
C_{\ell h}=O_{\ell h}V_{\ell h},
\qquad
\mathcal K_{\ell,s}(i,j;X)
=\sum_h a_{\ell h,ij}(s;X)C_{\ell h}(s).
$$

The precise question is:

> Which conditions on the task distribution, architecture, and initialization are
> necessary and sufficient for gradient flow to learn a task-aligned interaction
> kernel, and why does the learned kernel implement the required computation in a
> finite number of layers?

This is the upstream problem left open by fixed-kernel depth-dynamics theories. Those
theories analyze what a prescribed attention kernel does across layers. We ask why
training selects the correct kernel rather than a wrong or inaccessible one.

MQAR is the minimal one-step proof problem. LEGO is the next depth-composition problem.
Pythia-70M is an instrumentation calibration, not the source of a training law.

## 2. The exact single-layer object

For the complete matrix-MQAR population with three concepts and two memory items, the
48 episodes enumerate all ordered target--distractor pairs, memory orders, and binary
values. Define

$$
S=EQ^{\top}KE^{\top},
\qquad
g=w^{\top}Oz,
$$

and, for target concept $q$ and distractor $d$,

$$
a_{qd}=\frac{e^{S_{qq}}}{1+e^{S_{qq}}+e^{S_{qd}}},
\qquad
b_{qd}=\frac{e^{S_{qd}}}{1+e^{S_{qq}}+e^{S_{qd}}}.
$$

The delivered and target kernels are

$$
\kappa(\theta)=\bigl((ga_{qd},gb_{qd})\bigr)_{q\ne d},
\qquad
\kappa^*=\bigl((1,0)\bigr)_{q\ne d}.
$$

Complete value enumeration gives the exact identity

$$
R(\theta)=\frac1{12}\|\kappa(\theta)-\kappa^*\|_2^2.
$$

Thus, on this population, low risk and correct delivered-kernel learning are the same
mathematical statement. Attention mass alone is not the target: the value path $C=OV$
and readout must deliver the correct coefficient.

The observable $(S,g)$ determines the function but not an autonomous training flow.
The pullback metric depends on the raw factors and their conserved tensors. General
$GL$ gauge transformations preserve the function but are not Euclidean isometries, so
two functionally equivalent raw initializations need not follow the same gradient-flow
trajectory.

## 3. Established results

### 3.1 A positive theorem on a controlled branch

For the registered role-tied one-head reduction, let $a(\delta)$ and $b(\delta)$ be
the target and distractor masses, and define

$$
D(\delta)=a(\delta)-\frac{m-1}{m}b(\delta),
\qquad
h(g,\delta)=gD(\delta).
$$

If $m\ge2$, all six scalar factors are initially positive, and
$h(g_0,\delta_0)<1$, then

$$
\delta(s)\to+\infty,
\qquad
g(s)\to1,
\qquad
R(s)\to0.
$$

This proves that gradient flow can learn the correct softmax kernel. It is a genuine
existence theorem, not a theorem for unrestricted matrix factorization.

### 3.2 Exact gradients, factor invariants, and access

The full finite population yields closed-form gradients for $E,Q,K,O,z,w$. The exact
conserved tensors include

$$
I_{QK}=QQ^{\top}-KK^{\top},
\qquad
I_{EQK}=E^{\top}E-Q^{\top}Q-K^{\top}K,
$$

$$
J_L=ww^{\top}-OO^{\top},
\qquad
J_R=O^{\top}O-zz^{\top}.
$$

These identities constrain the raw flow, but they do not determine its task
orientation. Balance, full rank, nonzero initial gradient, and small norm are not
sufficient for success.

At the function level, $\nabla_{S,g}R$ is nonzero at every finite point of this
matrix-MQAR quotient. Raw-factor critical sets nevertheless exist because the
factor pullback can lose access to that nonzero task gradient.

### 3.3 The cumulative-access law

Let $e=\kappa-\kappa^*$ and $J_\kappa=D\kappa$. For $e\ne0$, define

$$
\alpha(\theta)
=\frac{\|J_\kappa(\theta)^{\top}e(\theta)\|_2^2}
{\|e(\theta)\|_2^2}.
$$

Population gradient flow satisfies the exact identity

$$
\dot R=-\frac13\alpha R,
\qquad
R(s)=R(0)\exp\left(-\frac13\int_0^s\alpha(\theta_t)\,dt\right).
$$

Therefore

$$
R(s)\to0
\quad\Longleftrightarrow\quad
\int_0^\infty\alpha(\theta_s)\,ds=\infty.
$$

If $R_\infty>0$, then cumulative task access is finite and exactly

$$
\int_0^\infty\alpha(\theta_s)\,ds
=3\log\frac{R(0)}{R_\infty}.
$$

This is the central law obtained so far. It converts kernel learning into a boundary
selection problem. It is not yet an initialization-only classifier.

### 3.4 Exact failure mechanisms

The unrestricted convergence claim is false.

| Failure mechanism | Certified limit | Meaning |
|---|---:|---|
| zero Q/K or collapsed concept contrast | finite wrong critical set | the quotient requests an update but factorization cannot transmit it |
| value-path death | $R=1/2$ | the correct value cannot reach the output |
| large-score self-only horn | $R_\infty=1/2$ | softmax saturation makes total task access finite |
| arbitrarily small open collapse cone | $R_\infty=1/4$ | target and distractor are both delivered with coefficient $1/2$ |

The small-initialization basin has positive lower density at the origin. It refutes
both “almost every regular initialization succeeds” and “failure probability vanishes
under small initialization.” The large-score and small-initialization proofs are
tracked in the proofs directory.

Correct retrieval also requires unbounded margins:

$$
R\to0
\quad\Longrightarrow\quad
S_{qq}-S_{qd}\to+\infty
\quad\text{for every }q\ne d.
$$

Hence a compact LaSalle argument in $(S,g)$ cannot prove convergence.

### 3.5 Partial singular-boundary classification

The latest exact proof program establishes the following modules, but not a complete
atlas.

- Three symmetry-related finite antisymmetric wrong families are strict saddles; their
  stable capture is ambient Lebesgue-null.
- At an accessible semidefinite first-score endpoint, tangent-dominant silent approach
  is excluded by the exact Gram-tail system. Inaccessible endpoints require
  $\det H=0$, an algebraic-null initial balance locus.
- If any ordered pair has an incorrect task margin at a finite state, then
  $R\ge1/24$. Thus $R<1/24$ is an all-future correct-margin chamber, although it does
  not by itself imply $R\to0$.
- The aligned rank-one flow has a coordinate-wall barrier $R>5/57$. Below this value,
  wall crossings are finite and the low-risk chamber order is fixed.
- A comparable-small-pair finite-logit boundary point is an angular saddle with
  $R\approx0.0851243$.
- A distinct one-log hierarchy has a locally attracting terminal tube on the balanced
  aligned rank-one invariant leaf, with $R_\infty\approx0.0634385$. Its stability in
  the full 42-dimensional raw system is not proved.

Two negative lessons are already exact. Static softmax support faces do not determine
the dynamics because finite exponential and logarithmic ratios matter inside a face.
The quadratic balance invariants do not determine the terminal outcome; oriented raw
factor information is necessary. A relevant hierarchy also has a divergent Lorentz
boost, so ordinary finite scattering coordinates are insufficient.

## 4. What the experiments establish

Experiments are evidence about the failure mechanisms above. They do not replace the
population-gradient-flow proof.

| Study | Design and observation | Valid conclusion |
|---|---|---|
| M1 access boundary | Public Zoology-compatible MQAR; four-layer, four-head, $d=128$ Transformer; 20 paired seeds. Exact $Q=K=0$ remains zero and gives accuracy $0.0721$ at $(L,m)=(64,4)$ and $0.0069$ at $(256,16)$. A $2^{-8}$ Q/K scale grows from norm $0.0283$ to $21.4935$ and reaches $0.9170$ and $0.8545$. | The bilinear zero-access boundary persists and is harmful in a standard model. Nonzero finite-step escape is possible. |
| M2 orientation | Same model and data; five paired arms; 20 seeds; 6400 AdamW steps. At $(256,16)$ all arm means lie in $[0.9504,0.9620]$; simultaneous signed contrasts include zero. Negative Q/K cosine moves from $-1$ to $0.0976$. | Raw initial sign is not a standalone boundary in a multilayer Transformer. The architecture repairs or leaves the reduced invariant branch. |
| Matrix ODE audit | Float64 DOP853 on the exact 48-episode population. One full-rank trajectory reduces risk from $0.2795$ to $0.00646$ by time 10 with invariant drift below $4.34\times10^{-16}$. | The equations and implementation agree. One trajectory is not a convergence theorem. |
| Phase II | Multi-seed controlled study with source verification and 20,000 whole-seed bootstrap resamples. Constant and cosine arms fail the stable-plateau conjunction. Dense direct improves strongly; rank-matched direct does not. | The earlier residual is not irreducible. The result supports a rank/function-class boundary candidate, not a pure factorization-conditioning claim. |
| Pythia-70M v4 | Eight audited checkpoints and four templates on one training trajectory. Maximum accuracy is $0.715$, minimum risk is $0.379$, and $S_{\mathrm{key}}\in[-5.878\times10^{-5},0.223]$. | The pipeline is numerically closed. Routing is weak, nonmonotone, and template-dependent; the proposed universal trajectory story is not supported. |

Additional boundaries are important:

- M1 and M2 do not establish long-context generalization; accuracy near length 1024 is
  only about $0.38$--$0.46$.
- The high-sample swap study preserves the dense-versus-rank direction, but its worst
  heavy-tail precision gate still fails at the largest audited budget. It cannot define
  a precise rare-collision law.
- The no-FFN localization study did not yield a complete valid module-attribution
  family. QK, OV, and suffix patches are overlapping local hybrid estimands, not an
  additive decomposition. No unique downstream compensator is established.
- Pythia checkpoints, templates, heads, and layers are repeated measurements from one
  pretraining trajectory. They are not independent samples.

## 5. Theory--experiment alignment

The current evidence supports a narrow, coherent picture.

1. **Training can learn the correct interaction kernel.** The reduced positive theorem
   proves it, the exact ODE reproduces it, and M1 shows escape from small nonzero access.
2. **Representability is not enough.** Open wrong basins exist even when the target
   kernel is representable and the raw initialization is full rank and nondegenerate.
3. **The decisive object is cumulative access.** Failure occurs when factor dynamics or
   softmax saturation make the integral of $\alpha$ finite.
4. **A raw sign rule is too crude.** M2 shows that standard architecture can rotate away
   from $K=-Q$. The final condition must use data-defined delivered-kernel geometry and
   the full raw factor dynamics.
5. **Fixed-kernel depth dynamics is downstream of an unresolved selection theorem.**
   The repository verifies a prescribed-kernel clustering baseline and an exact local
   LEGO transition/composition bound, but it does not yet prove that training learns the
   required LEGO parent edges.

## 6. The remaining theorem

The next theorem is a necessary-and-sufficient initialization classification, not a
new architecture or a wider experiment grid. It must convert the trajectory identity

$$
\int_0^\infty\alpha(\theta_s)\,ds=\infty
$$

into conditions derived from the MQAR data law, the initialization, the oriented raw
factors, and their invariants.

The unresolved gates are:

1. **Rank-one exhaustion:** classify every finite critical direction and every relevant
   finite-logit or infinity chart.
2. **Semidefinite transfer:** construct a uniform center-stable lamination, or an
   equivalent negative-link atlas, for accessible semidefinite endpoints.
3. **Raw nonlinear transfer:** lift the reduced hierarchy tubes and separators to the
   full 42-dimensional factor flow with uniform remainder control.
4. **Terminal classifier:** prove a checkable rule that assigns each regular
   initialization to the task kernel or to a certified wrong kernel. Quadratic balance
   tensors alone cannot be that rule.
5. **Measure transfer:** determine which stable sets have zero or positive probability
   under the actual small random initialization law.

Only after these five gates close should the project prove the LEGO statement:

$$
\text{gradient flow learns both required parent edges}
\quad\Longrightarrow\quad
\text{the learned local operator composes correctly through depth}.
$$

## 7. Fixed claim boundary and next action

The project does **not** currently claim a universal clustering theorem, a general law
of Pythia training, a new theorem that low rank exists, an identified QK/OV/FFN
compensator, or a complete singular-boundary classification.

The immediate work is theory-facing: finish the rank-one and semidefinite boundary
atlas, then either prove the full terminal classifier or extract an exact new
full-MQAR counterexample. No new model family is needed for this step. Existing M1,
M2, Phase-II, Pythia, and LEGO artifacts remain preserved as boundary tests and
downstream validation targets.

Detailed sources:

- [Single-layer theorem boundary](SINGLE_LAYER_THEORY_STATUS.md)
- [Exact matrix-MQAR gradients and obstructions](MATRIX_MQAR_C3M2_RESULT.md)
- [Boundary-selection result](MATRIX_MQAR_BASIN_RESULT.md)
- [Large-norm wrong-basin proof](../proofs/MATRIX_MQAR_LARGE_NORM_COUNTEREXAMPLE_PROOF.md)
- [Small-initialization wrong-basin proof](../proofs/MATRIX_MQAR_SMALL_INIT_COUNTEREXAMPLE_PROOF.md)
- [M1 result](MQAR_M1_BOUNDARY_RESULT.md)
- [M2 result](MQAR_M2_ORIENTATION_RESULT.md)
- [Experiment positioning](EXPERIMENT_POSITIONING.md)
- [Literature map](LITERATURE_MAP.md)
