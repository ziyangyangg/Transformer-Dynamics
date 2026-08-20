# Method specification

## 1. Target theorem

For \((X,Y)\sim\mathcal D\),

\[
R(\theta)=\mathbb E_{\mathcal D}\,\ell(f_\theta(X),Y),
\qquad
\dot\theta_s=-\nabla_\theta R(\theta_s).
\]

The primary objective is to prove, under explicit data, initialization, and scale
assumptions, that training creates a task-aligned interaction kernel and that the
resulting finite-depth network implements the task interaction graph \(G^*(X)\).

The theorem must control three quantities:

\[
\gamma_s(X)
=u_{iJ^*}(s)-\max_{j\ne J^*}u_{ij}(s),
\]

\[
\mathcal E_{\rm transport}(s)
=\left\|\mathcal K_s(i,J^*;X)z_{J^*}-m_i^*(X)\right\|
+\sum_{j\ne J^*}\left\|\mathcal K_s(i,j;X)z_j\right\|,
\]

\[
\mathcal E_{\rm depth}(s,L)
=\left\|\Phi_{\theta_s}^{L}(X)-\Phi^*(X)\right\|.
\]

Risk reduction without these structural bounds is insufficient.

## 2. Minimal identifiable task

The theorem-bearing task is a public deterministic generator, not an empirical claim
about natural language. Each episode contains \(m\) distinct concept-value pairs,

\[
(c_1,v_1),\ldots,(c_m,v_m),q,
\qquad
v_i\overset{\rm iid}{\sim}\mathrm{Unif}\{-1,+1\},
\]

\[
J\sim\mathrm{Unif}\{1,\ldots,m\},
\qquad q=c_J,\qquad Y=v_J.
\]

Fresh random values prevent the concept embedding from memorizing labels. The known
interaction graph contains one required edge, query \(\to J\). Position alone cannot
solve the task because \(J\) changes independently across episodes.

This generator is used because \(G^*(X)\), the population law, and all counterfactuals
are exact. Public algorithmic/state-tracking tasks may later test generalization, but
they do not replace the minimal proof instance.

## 3. Model and learned kernel

The active toy model is a finite causal pre-normalized Transformer with learned
representations, factorized \(Q/K/O/V\), exact softmax, residual connections, and a
trained readout. For layer \(\ell\), head \(h\),

\[
B_{\ell h}=Q_{\ell h}^{\top}K_{\ell h},
\qquad
C_{\ell h}=O_{\ell h}V_{\ell h},
\]

\[
a_{\ell h,ij}
=\frac{\exp\{(z_i^\ell)^\top B_{\ell h}z_j^\ell/\sqrt{d_h}\}}
{\sum_{k\le i}\exp\{(z_i^\ell)^\top B_{\ell h}z_k^\ell/\sqrt{d_h}\}},
\]

\[
\mathcal K_{\ell}(i,j;X)
=\sum_h a_{\ell h,ij}(X)C_{\ell h},
\qquad
m_i^\ell=\sum_{j\le i}\mathcal K_{\ell}(i,j;X)z_j^\ell.
\]

\(B\) chooses sources; \(C\) transports their content into the residual stream. Raw
\(Q,K,V,O\) factors are gauge-dependent, so primary statements use \(B,C\) or function
values.

## 4. Training method

The theory uses population gradient flow. Exact enumeration is required whenever the
support is feasible; otherwise fresh counter-addressable batches approximate the same
expectation. Experimental optimizers are robustness checks, not identities with
gradient flow.

Every trajectory records:

- complete model, optimizer, scheduler, and random-stream state;
- \(R,B,C,\gamma,\mathcal E_{\rm transport}\) at fixed checkpoints;
- immutable configuration, source hashes, and failure ledger;
- independent training seeds as the only inferential units.

## 5. Identifiability and interventions

The output value coefficient is measured by value flips or exact Walsh coefficients.
Direct source dependence is measured by blocking each query-to-memory edge, renormalizing
softmax, and recomputing all descendants.

These interventions answer different questions:

- low risk and Walsh leakage test functional use of the correct value;
- slot blocking tests a registered direct path;
- neither proves that a particular head or module is uniquely responsible.

Low risk implies positive blocking selectivity only under explicit value-path
identifiability, gain-sign, and no-bypass assumptions. Signed cancellation provides a
zero-risk counterexample without those assumptions.

## 6. Training-to-depth bridge

At frozen training time \(s\), the learned kernel is inserted into the layer recursion.
The proof must derive softmax leakage from \(\gamma_s\), propagate transport error
through residual layers, and bound \(\mathcal E_{\rm depth}(s,L)\). Training time \(s\)
and layer depth \(\ell\) are never identified.

## 7. Evidence hierarchy

1. Exact theorem or counterexample.
2. Multi-seed controlled evidence for theorem quantities.
3. Single-trajectory checkpoint evidence, explicitly descriptive.
4. Local diagnostics, used only to find assumptions or numerical failures.

Low-rank attention, nonorthogonal embeddings, fixed-kernel clustering, attention maps,
and local patch effects are prior knowledge or diagnostics. They are not contributions
by themselves.

## 8. Stop rules

No new model family, dataset, grid, or diagnostic is added unless it tests a variable in
Sections 1–6. A failed gate is reported and stopped. Pythia checkpoints are not seeds.
Module attribution is not claimed without a common-base closed decomposition. Rare
collisions remain a possible mechanism only if high-precision replicated evidence
survives and known kernel geometry fails to explain them.
