# P39 source-integrity contract

Status: frozen on 2026-08-20 before any prospective seed 300--303 was
generated or inspected.

P39 may consume a population-gradient-flow directory only after the loader
reconstructs the evidence below. A success marker or manifest field is never
accepted as scientific evidence by itself.

## 1. Study identity

For an original study, the recorded hash must equal

\[
h_{\rm cfg}=\operatorname{SHA256}_{\rm canonical}
  (\mathrm{PopulationGFStudyConfig}).
\]

For a numerical remedy, it must instead equal the canonical hash of the complete
refinement identity, including the source fingerprint, original step, refinement
factor, and refined step. The embedded refined config must exactly equal the
recorded study config.

## 2. Exact tabular identity

The JSON trajectory is the typed source. Canonical CSV serialization of every
JSON row, in the same order and with sorted columns, must be byte-for-byte equal
to the CSV trajectory. This rejects truncation, row permutation, and rounded
exports.

## 3. Full-grid and checkpoint linkage

For every normalized divisor \(q\in\{1,2,4\}\), every aligned index \(i\)
must satisfy

\[
k_{i,q}=i\,q\,a,\qquad
s_i=i\,a\,\eta_0,\qquad
\Delta s_q=\eta_0/q,
\]

where \(a\) is the alignment stride. Root rows must exactly equal the
corresponding per-divisor rows. The continuation must contain those same rows,
the same initialization hash, and the registered final fine step. Reloading its
model state and remeasuring the complete population must reproduce the final P37
row.

## 4. Initial Hessian and P36

The initializer and complete finite population are reconstructed from the config.
The exact dense symmetric Hessian is recomputed. Its largest and smallest
eigenvalues, residual, method, and parameter count must agree with the stored
initial-Hessian record. Then

\[
\eta_{\rm P36}
=\min\!\left(0.003,\frac{0.25}{\lambda_{\max}+10^{-12}}\right)
\]

must reproduce the original step. A refinement with factor \(r\) must use
\(\eta_0=\eta_{\rm P36}/r\).

## 5. P37 identities and recomputed P38

Every scalar is finite. At every observation,

\[
\Xi_{\rm value}-K_{\rm target}
=\mathrm{flip\_walsh\_identity\_gap},
\]

and the recorded Parseval gap must equal

\[
2R-E_T-L_D-L_H-L_0.
\]

For each P37 coordinate \(z_j\), the loader recomputes

\[
D_j(q,2q)=
\frac{\lVert z_j^{(q)}-z_j^{(2q)}\rVert_2}
     {\lVert z_j^{(2q)}-z_j^{(2q)}(0)\rVert_2+10^{-12}}
\]

for both nested pairs. P38 passes only when every recomputed discrepancy is at
most 0.10. Stored comparisons, failed-coordinate lists, and manifest decisions
must match this recomputation.

## 6. Provenance and P39 replay

P39 binds SHA-256 hashes of all root artifacts, all three per-divisor manifests
and trajectories, and all three continuation checkpoints. It also binds the
protocol and measurement implementation files. A committed P39 fast path
reconstructs every JSON/CSV output from the current immutable sources and requires
byte identity before returning a skipped result.

Two non-gating sensitivities are mandatory:

1. refit after dropping duplicate coordinate \(\Xi_{\rm value}\), retaining
   independently measured \(K_{\rm target}\);
2. report whether all seeds share one actual finest Euler divisor. If not, a
   common-finest-resolution sensitivity is explicitly unavailable and requires a
   separate prospectively fixed cohort.

## 7. Current stopping rule

Numerical-validation seeds 201 and 202 failed the final frozen
\((\eta/16,\eta/32,\eta/64)\) triplet. Therefore fresh seeds 300--303 are not
generated or inspected, the P39 vector field is not fit, and no closure or
nearest-neighbor counterexample claim is made.
