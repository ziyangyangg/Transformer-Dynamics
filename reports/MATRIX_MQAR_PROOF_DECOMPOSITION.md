# Matrix MQAR Boundary Selection: Formal Proof Decomposition

> **Superseded target.** Theorem T in this document is false. A verified large-norm
> open basin and a verified positive-density small-initialization basin converge to
> wrong kernels. Use [SINGLE_LAYER_THEORY_STATUS.md](SINGLE_LAYER_THEORY_STATUS.md)
> for the active success/failure classification problem. This document remains only
> the historical lemma decomposition.

## 1. Purpose and scope

This document turns the active research question into proof-sized mathematical
contracts.  It does not assume the desired basin, aligned attention, $K=Q$, a
uniform pullback constant, or bounded quotient variables.  Every failed contract
must return either a counterexample or the exact additional condition that it shows
to be necessary.

The frozen order is

$$
\text{matrix MQAR boundary selection}
\Longrightarrow
\text{general MQAR capacity/access theorem}
\Longrightarrow
\text{LEGO local-kernel learning}
\Longrightarrow
\text{depth composition}.
$$

Only the first implication is active.  No new model, head, layer, normalization, or
dataset is part of this problem.

## 2. Frozen finite problem

Let $C=d=3$, $m=2$, and

$$
\Omega
=
\left\{
(c_1,c_2,J,v_1,v_2):
c_1\ne c_2,
\ J\in\{1,2\},
\ v_1,v_2\in\{-1,+1\}
\right\}
$$

with the uniform law.  The query and label are $q=c_J$ and $Y=v_J$.
The parameter is

$$
\theta=(E,Q,K,O,V,w),
\qquad
E,Q,K,O,V\in\mathbb R^{3\times3},
\quad
w\in\mathbb R^3,
$$

and $u=e_1$ is fixed.  Define

$$
S=EQ^\top K E^\top,
\qquad
g=w^\top OVu.
$$

For an ordered target--distractor pair $(q,d)$, $q\ne d$, let

$$
D_{qd}=1+e^{S_{qq}}+e^{S_{qd}},
\qquad
a_{q\mid qd}=\frac{e^{S_{qq}}}{D_{qd}},
\qquad
b_{d\mid qd}=\frac{e^{S_{qd}}}{D_{qd}}.
$$

The extra denominator term is a query-self token with fixed score and value zero.
The model output and population risk are

$$
f_\theta=g\bigl(a_{q\mid qd}v_q+b_{d\mid qd}v_d\bigr),
\qquad
R(\theta)=\frac12\mathbb E_\Omega(f_\theta-Y)^2.
$$

There are $N=C(C-1)=6$ ordered pairs.  Introduce the delivered coefficients

$$
x_{qd}(\theta)=g a_{q\mid qd},
\qquad
y_{qd}(\theta)=g b_{d\mid qd},
\qquad
\kappa(\theta)=\bigl((x_{qd},y_{qd})\bigr)_{q\ne d},
$$

and the task kernel

$$
\kappa^*=\bigl((1,0)\bigr)_{q\ne d}.
$$

The exact quotient risk is

$$
\mathcal R(S,g)
=
\frac1{2N}\sum_{q\ne d}
\left[(x_{qd}-1)^2+y_{qd}^2\right],
\qquad
R(\theta)=\mathcal R\bigl(S(\theta),g(\theta)\bigr).
\tag{1}
$$

Gradient flow is

$$
\dot\theta_s=-\nabla_\theta R(\theta_s).
\tag{2}
$$

### Gauge equivalence

For $A,H,T\in\mathrm{GL}(3)$ and $\alpha\ne0$, the following raw-factor
transformations preserve the represented function:

$$
(Q,K)\mapsto(AQ,A^{-\top}K),
\qquad
(O,V)\mapsto(OH,H^{-1}V),
$$

and the embedding-coordinate and value-scale transformations can be lifted as

$$
(E,Q,K)\mapsto(ET^\top,QT^{-1},KT^{-1}),
\qquad
(w,O)\mapsto(\alpha^{-1}w,\alpha O).
$$

These maps leave $(S,g)$ unchanged. Their compositions generate the registered
functional equivalence used below. Consequently, a theorem must concern the
delivered kernel $\kappa$ or the quotient $(S,g)$, not a raw representative.

## 3. Correct target and a necessary correction

Because the population is finite, (1) gives

$$
R(\theta_s)\to0
\quad\Longleftrightarrow\quad
\max_{q\ne d}
\max\{|x_{qd}(\theta_s)-1|,|y_{qd}(\theta_s)|\}
\to0.
\tag{3}
$$

Equation (3), rather than any raw factor limit, is the task-identifiable target.
It implies

$$
S_{qq}(s)-S_{qd}(s)
=
\log\frac{a_{q\mid qd}(s)}{b_{d\mid qd}(s)}
\to+\infty
\qquad(q\ne d).
\tag{4}
$$

It does **not** imply $g(s)\to1$.  For example, for $n\ge3$, set

$$
g_n=n,
\qquad
a_n=\frac1n,
\qquad
b_n=\frac1{n^2},
\qquad
p_n^0=1-a_n-b_n,
$$

and choose every diagonal and off-diagonal score as

$$
S_{qq}^{(n)}=\log\frac{a_n}{p_n^0},
\qquad
S_{qd}^{(n)}=\log\frac{b_n}{p_n^0}
\quad(q\ne d).
$$

Then $x_{qd}=1$, $y_{qd}=1/n\to0$, and $R\to0$, while $g_n\to\infty$.
Thus selection of $g=1$ is a separate implicit-bias problem and is not required for
kernel learning.

## 4. Candidate theorem to prove or refute

Define the finite regular-access set

$$
\Theta_{\rm reg}
=
\left\{
\theta:
\det E\,\det Q\,\det K\ne0,
\quad
c_g(\theta)>0
\right\},
\tag{5}
$$

where

$$
c_g(\theta)
=
\|OVu\|^2
+\|w\|^2\|Vu\|^2
+\|O^\top w\|^2\|u\|^2.
\tag{6}
$$

Let $\lambda_\Theta$ denote Lebesgue measure on parameter space.

**Theorem T (maximal minimal-model target).** There is a
$\lambda_\Theta$-null set $\mathcal N\subset\Theta_{\rm reg}$ such that, for
every $\theta_0\in\Theta_{\rm reg}\setminus\mathcal N$, the solution of (2)
exists for all $s\ge0$ and satisfies

$$
\max_{q\ne d}
\max\{|x_{qd}(\theta_s)-1|,|y_{qd}(\theta_s)|\}\to0.
\tag{T}
$$

Consequently, $R(\theta_s)\to0$ and all six directed margins in (4) diverge.

Equivalently, (T) holds with probability one for every initialization law
$\nu\ll\lambda_\Theta$ satisfying $\nu(\Theta_{\rm reg})=1$.
The theorem is deliberately a prove-or-refute target.  A positive-measure regular
basin that violates (T) refutes it and must be retained as the missing-condition
witness.

## 5. Exact pullback objects

Let

$$
G_S=\nabla_S\mathcal R(S,g),
\qquad
\gamma=\partial_g\mathcal R(S,g),
\qquad
G_B=E^\top G_SE.
$$

The exact parameter gradients are

$$
G_E=G_SEB^\top+G_S^\top EB,
\qquad
G_Q=KG_B^\top,
\qquad
G_K=QG_B,
\tag{7}
$$

$$
G_w=\gamma OVu,
\qquad
G_O=\gamma w(Vu)^\top,
\qquad
G_V=\gamma O^\top w u^\top.
\tag{8}
$$

For the quotient map $\Psi(\theta)=(S(\theta),g(\theta))$, define the
gradient-specific access ratio

$$
\alpha_\Psi(\theta)
=
\frac{
\|D\Psi(\theta)^*\nabla\mathcal R(\Psi(\theta))\|^2
}{
\|\nabla\mathcal R(\Psi(\theta))\|^2
}
\quad
\text{when }\nabla\mathcal R\ne0.
\tag{9}
$$

The finite factor-access singular set is

$$
\Sigma_{\rm fin}
=
\left\{
\theta:
\nabla\mathcal R(\Psi(\theta))\ne0,
\quad
D\Psi(\theta)^*\nabla\mathcal R(\Psi(\theta))=0
\right\}.
\tag{10}
$$

Risk dissipation is exactly

$$
\dot R
=
-\|G_E\|_F^2-\|G_Q\|_F^2-\|G_K\|_F^2-c_g\gamma^2
=
-\|D\Psi^*\nabla\mathcal R\|^2.
\tag{11}
$$

Equation (9) is a diagnostic, not an assumption.  A uniform lower bound
$\alpha_\Psi\ge\mu>0$ may be proved on a derived invariant region, but it may not be
inserted into Theorem T as a premise.

## 6. Proof dependency graph

The logical order is

$$
\mathrm{L0}\longrightarrow\mathrm{L2},
\qquad
\mathrm{L1}\longrightarrow\mathrm{L3},
\qquad
(\mathrm{L1},\mathrm{L2})\longrightarrow\mathrm{L4},
$$

$$
(\mathrm{L3},\mathrm{L4})\longrightarrow\mathrm{L5},
\qquad
\mathrm{L1}\longrightarrow\mathrm{L6},
\qquad
(\mathrm{L2},\mathrm{L3},\mathrm{L4},\mathrm{L6})
\longrightarrow\mathrm{L7},
$$

$$
(\mathrm{L5},\mathrm{L7})\longrightarrow\mathrm{L8}
\longrightarrow\mathrm{L9}=\mathrm{T}.
$$

L0 and L1 are algebraic foundations.  L2 makes infinity analyzable.  L3 and L4
list every possible wrong endpoint.  L5 decides whether those endpoints attract.
L6 prevents finite-time escape.  L7 proves that the list is exhaustive along actual
trajectories.  L8 turns local instability into an almost-everywhere statement.

## 7. Proof contracts

### L0. Observable target boundary

**Statement.** Prove (1), (3), and (4), and classify all zero-risk asymptotic
sequences in the delivered coordinates $\kappa$.  Prove explicitly that $g$ is not
identified by the task.

**Role.** Fixes the conclusion before any dynamical argument.  It prevents a proof
of an internal gauge or gain limit from being mistaken for kernel learning.

**Failure output.** Any additional functionally distinct zero-risk kernel would
show that the data law is not identifiable and would stop the program.

**Status.** The risk identity, margin implication, and the $g_n\to\infty$ witness
above are established.

### L1. Quotient pullback and finite access

**Statement.** Verify (7)--(11).  Prove that
$\Sigma_{\rm fin}\cap\Theta_{\rm reg}=\varnothing$.  Determine the exact ranks and
null spaces of $D\Psi(\theta)$ when $E,Q,K$ or the value path lose rank.

**Role.** Separates a quotient request from the factors' ability to execute it.
This is the precise form of the factor-access question.

**Failure output.** A regular point in $\Sigma_{\rm fin}$ refutes the proposed
regular-access definition (5) and supplies a missing algebraic condition.

**Status.** The chain-rule identities are established; the rank-stratified null-space
classification is open.

### L2. Boundary-regular compactification

For a risk sublevel $R\le r_0$, (1) implies

$$
|x_{qd}|\le1+\sqrt{2Nr_0},
\qquad
|y_{qd}|\le\sqrt{2Nr_0}.
\tag{12}
$$

Define

$$
r_g=\frac{g}{\sqrt{1+g^2}},
\qquad
p^0_{qd}=1-a_{q\mid qd}-b_{d\mid qd}.
$$

These variables obey

$$
\sqrt{1-r_g^2}\,x_{qd}=r_g a_{q\mid qd},
\qquad
\sqrt{1-r_g^2}\,y_{qd}=r_g b_{d\mid qd}.
\tag{13}
$$

**Statement.** Construct a compact state $\overline{\mathcal X}$ containing the
closure of every finite-risk orbit, including compatible attention probabilities,
delivered coefficients, factor directions, and required scale variables.  Find a
positive interior time change for which the induced vector field extends at least
$C^1$ to every boundary stratum used in L4--L8.  Identify the correct face as

$$
\mathcal F_*=\{x_{qd}=1,\ y_{qd}=0\text{ for every }q\ne d\}.
\tag{14}
$$

**Role.** Replaces the false bounded-$(S,g)$ LaSalle route.  Without L2, limits at
diverging margins cannot be classified.

**Failure output.** Record the precise scale or direction missing from the proposed
compactification; do not assume bounded raw parameters.

### L3. Finite wrong critical strata

Define

$$
\mathcal Z_{\rm fin}
=
\{\theta:\nabla_\theta R(\theta)=0,\ R(\theta)>0\}/
(\text{gauge}\times S_3).
\tag{15}
$$

**Statement.** Decompose

$$
\mathbb R^3=\operatorname{span}\{\mathbf1\}\oplus\mathbf1^\perp.
$$

Stratify $E,Q,K$ by the ranks and relative orientations of their
trivial and contrast restrictions and stratify the value path by $c_g$.  Solve (15)
on every stratum.  The output must be a finite list of connected analytic families,
or a proof that a further continuous modulus is necessary.  Transcendental roots
must be isolated with interval certificates rather than floating-point guesses.

**Role.** Produces the complete list of finite mechanisms by which training can stop
while the task kernel is still wrong.

**Failure output.** A previously unknown regular positive-risk local minimum is an
immediate counterexample candidate for Theorem T.

### L4. Wrong invariant strata at infinity

Let $\bar F$ be the compactified vector field from L2.  Define

$$
\mathcal Z_\infty
=
\left\{
z\in\partial\overline{\mathcal X}:
R(z)>0,
\ z\text{ belongs to a compact }\bar F\text{-invariant set}
\right\}.
\tag{16}
$$

**Statement.** Classify every invariant component of (16), including rank collapse,
value-path death, wrong margin rays, and their intersections.  Prove that the list is
closed under taking boundary limits of its own strata.

**Role.** Finite critical-point classification alone cannot exclude a trajectory
whose norms diverge while its delivered kernel remains wrong.

**Failure output.** An unclassified recurrent set or wrong asymptotic ray blocks
Theorem T and becomes a new explicit subproblem.

### L5. Normal stability of every wrong stratum

**Statement.** For each component $Z$ of
$\mathcal Z_{\rm fin}\cup\mathcal Z_\infty$, remove gauge and stratum-tangent
directions and compute the normal linearization

$$
L_z=\Pi_{N_z}D\bar F(z)|_{N_z}.
\tag{17}
$$

Prove one of the following, uniformly on each compact component:

1. $L_z$ has a nontrivial unstable bundle and $W^{cs}(Z)$ has codimension at least
   one;
2. higher-order center-manifold terms still give a measure-zero attracting set; or
3. $Z$ has a positive-measure basin, which refutes Theorem T.

**Role.** Converts a list of wrong solutions into a basin statement.  The existence
of a wrong stationary point alone does not refute an almost-everywhere theorem.

**Known witness.** The canonical uniform wrong boundary has $R=1/4$ and at least
four exact unstable contrast modes with growth rate $e_\perp^2/8$.

### L6. Global existence and normalized noncollapse

The flow preserves the exact balancing quantities

$$
QQ^\top-KK^\top,
\qquad
\|E\|_F^2-2\|Q\|_F^2,
\qquad
\|E\|_F^2-2\|K\|_F^2,
\tag{18}
$$

$$
O^\top O-VV^\top,
\qquad
\|w\|^2-\|O\|_F^2,
\qquad
\|w\|^2-\|V\|_F^2.
\tag{19}
$$

**Statement.** Starting only from (1), (7)--(8), and (18)--(19), prove that every
solution from $\Theta_{\rm reg}$ exists for all finite $s$.  Prove precompactness of
the normalized factor variables used in L2.  Determine whether a task contrast can
lose factor access asymptotically; if it can, identify the corresponding L3/L4
stratum.

**Role.** Makes every later $\omega$-limit argument legitimate and prevents hidden
finite-time factor blow-up.

**Failure output.** A finite-time blow-up or an unclassified access-collapse path
refutes the current theorem formulation.  It may not be repaired by simply assuming
bounded trajectories.

### L7. Exhaustive $\omega$-limit classification

**Statement.** Use (11), L2, and L6 to prove that every compactified $\omega$-limit
set satisfies

$$
\omega(\theta_0)
\subseteq
\mathcal F_*
\cup\mathcal Z_{\rm fin}
\cup\mathcal Z_\infty.
\tag{20}
$$

Rule out unclassified cycles, heteroclinic chains through infinitely many rank
strata, and recurrent sets created only by the time change.

**Role.** Establishes that L3--L5 did not omit a global escape mechanism.

**Failure output.** Any additional invariant set must be added to L4 before the
proof can continue.

### L8. Almost-everywhere exclusion of wrong boundaries

**Statement.** Prove

$$
\lambda_\Theta\left(
\left\{
\theta_0\in\Theta_{\rm reg}:
\omega(\theta_0)\cap
(\mathcal Z_{\rm fin}\cup\mathcal Z_\infty)\ne\varnothing
\right\}
\right)=0.
\tag{21}
$$

The proof must justify countability or finite stratification and must verify every
hypothesis of the stable/center-stable manifold result it invokes.

**Role.** This is the exact boundary-selection step.  It converts local repulsion
into the random-initialization conclusion.

**Failure output.** A positive-measure wrong basin is a rigorous counterexample to
Theorem T.  The missing condition must then be read from that basin, not invented.

### L9. Close Theorem T

**Statement.** Combine L0 and L8 to prove (T), or present the exact counterexample
returned by L3--L8.  State the smallest condition supported by the counterexample
analysis if the maximal theorem is false.

**Role.** This is the first training-aware kernel-selection theorem.  It explains
how the task law and factorized gradient flow select the delivered interaction
kernel; it does not yet claim a depth-composition theorem.

### L10. Optional implicit bias inside the correct face

**Statement.** Conditional on L9, classify which task-equivalent zero-risk ray is
selected: finite $g_*>1$, $g_*=1$, $g\to\infty$, or nonconvergent $g$.

**Role.** This can explain internal gain/attention allocation but is not needed for
correct retrieval and is not a prerequisite for general MQAR or LEGO.

## 8. Known counterexamples that every proof must survive

1. **Zero-factor barrier:** $Q=K=0$ can block a nonzero quotient score gradient.
2. **Balanced wrong orientation:** the permutation-symmetric branch
   $E=I$, $Q=\alpha I$, $K=-\alpha I$ is full rank and balanced but cannot make
   both directed margins of a concept pair positive.
3. **Wrong saddle:** a positive-risk access-singular boundary may exist yet have only
   a measure-zero stable set; its existence alone does not refute Theorem T.
4. **Gain non-identifiability:** the sequence in Section 3 learns the exact delivered
   kernel with $g\to\infty$.

## 9. Rethlas execution contract

The Rethlas problem file is

```text
PROBLEM_FILE=data/transformer_dynamics/matrix_mqar_boundary_selection.md
```

Rethlas must not be asked for a single unstructured proof.  It should process the
contracts in dependency order:

1. independently verify L0 and L1;
2. solve L2 before making any statement about limits at infinity;
3. solve L3 and L4 before invoking stable-manifold theory;
4. run counterexample search on every failed L2--L7 lemma;
5. attempt L8 and L9 only after the earlier artifacts are verified.

For each contract, the required output is: exact statement, proof or counterexample,
dependencies used, unresolved hypotheses, and the consequence for Theorem T.  A
numerical trajectory may falsify a claim, but it cannot certify one.

## 10. Relation to the final research program

If Theorem T is true, it supplies the missing training-time map

$$
(\mathcal D,R,\theta_0)
\longmapsto
\kappa(\theta_s)\to\kappa^*
$$

for the smallest learned-dictionary, factorized-QK/OV, exact-softmax retrieval
problem.  General MQAR then asks how representability and access depend on
$(C,m,d,d_h,H)$.  LEGO is subsequent: first prove that training learns each required
two-parent local kernel, then compose the already learned local errors through depth.

If Theorem T is false, the attracting counterexample is itself the result: it gives
the exact data/model/initialization condition missing from training-aware Transformer
dynamics.  In either case, L0--L9 resolve a necessary question before any larger-model
experiment can have a theorem-facing interpretation.
