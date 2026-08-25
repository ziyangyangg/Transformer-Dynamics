# Implementation Plan: Matrix MQAR Boundary Selection

This is the only active plan. The detailed theorem and literature dependencies are in
[`reports/MAIN_THEOREM_ROADMAP.md`](../reports/MAIN_THEOREM_ROADMAP.md).
The proof-sized statements and their dependencies are in
[`reports/MATRIX_MQAR_PROOF_DECOMPOSITION.md`](../reports/MATRIX_MQAR_PROOF_DECOMPOSITION.md).

## Fixed success criterion

For the complete $C=3,m=2,d=3$ MQAR population and the frozen one-head,
exact-softmax, factorized model, prove or refute

$$
\Pr_{\theta_0\sim\nu}\left[
\max_{\omega\in\Omega}\max_{i\in\{1,2\}}\left|g(s)a_i(s;\omega)-\mathbf 1\{i=J\}\right|\to0
\right]=1,
$$

where $\nu$ is an explicitly stated continuous random initialization with full score
and value access almost surely. The basin must be derived, not assumed.

The task does not identify $g$ separately. Target-kernel convergence can occur with
$g\to\infty$; a claim about $g\to1$ is an optional implicit-bias problem, not part
of the fixed success criterion.

This is a condition-discovery theorem. A pullback constant may be used
only as one sufficient certificate for factor access; it cannot replace the derivation of the
actual basin or be assumed uniformly positive.

## Dependency-ordered gates

### Gate A: exact functional problem — complete

- [x] Freeze the 48-episode population, model, gauges, and target kernel.
- [x] Verify exact factor gradients against enumeration and autograd.
- [x] Prove $2R$ equals target-kernel transport error.
- [x] Prove that correct retrieval forces every directed score margin to diverge.
- [x] Refute bounded-quotient LaSalle and unrestricted balanced initialization.

### Gate B: compactified boundary — active

- [ ] Express the risk on the attention simplex and classify its zero-risk face.
- [ ] Separate margin magnitude from margin direction.
- [ ] Derive a boundary-regular compactified flow, with time rescaling if required.
- [ ] Prove that normalized finite-risk trajectories have nonempty limit sets.

### Gate C: exhaustive singular-set classification — active after B.1

- [ ] Decompose the $C=3$ concept space into trivial and contrast $S_3$ modes.
- [ ] Stratify score factors by rank and relative orientation; stratify value access.
- [ ] Solve every parameter-stationary family modulo gauge and concept permutation.
- [ ] Certify exhaustiveness symbolically and with interval-checked witnesses.

### Gate D: local boundary selection — depends on C

- [x] Certify four unstable normal modes at the canonical uniform wrong boundary.
- [ ] Remove gauge-tangent directions from every P2 stratum.
- [ ] Prove an unstable normal mode for each wrong stratum or produce an attracting
  counterexample.
- [ ] Bound each wrong center-stable set under the initialization law.

### Gate E: global boundary selection — depends on B and C

- [ ] Prove global existence and normalized-factor precompactness.
- [ ] Derive integrated task-direction factor access from invariants and initialization.
- [ ] Show every limit set lies on the correct face or a classified wrong stratum.
- [ ] Exclude escape through a descending chain of access-singular strata.

### Gate F: minimal theorem — depends on B--E

- [ ] Combine compactification, stability, and global exclusion into the
  almost-everywhere theorem.
- [ ] State one exact counterexample for every necessary hypothesis.
- [ ] Re-run float64 adaptive ODE only as an adversarial audit of proved identities.

### Gate G: general MQAR — locked until F

- [ ] Derive the minimum task-dependent score/value rank for general $C,m,d$.
- [ ] Extend the singular-mode analysis from $S_3$ to $S_C$.
- [ ] Prove the finite-sample/SGD bridge without changing the task family.

### Gate H: LEGO depth composition — locked until F

- [ ] Derive the two-parent local kernel and its factorized learning law.
- [ ] Compose learned routing error and local-operator error through depth.
- [ ] Separate the result from the published LEGO learnability theorem.

## Stop rules

- Do not add a model family, dataset, head, layer, FFN, or normalization before Gate F.
- Do not assume aligned attention, $K=Q$, exact symmetry, or uniform pullback access.
- Do not interpret numerical trajectories when the adaptive-step audit fails.
- If a wrong stratum has a positive-measure basin, stop and state the missing condition.
