# Literature, Mathematical Tools, and Open-Theorem Map

Last source audit: 2026-08-25. The dependency-level reading list is maintained in
[the main theorem roadmap](MAIN_THEOREM_ROADMAP.md).

## Target joint statement

The project studies

$$
(\mathcal D,R,\theta_0)
\xrightarrow{\text{gradient flow}}
\{B_{\ell h}(s),C_{\ell h}(s)}
\xrightarrow{\text{exact softmax}}
\mathcal K_{\ell,s}
\xrightarrow{\text{depth}}
\Phi_{\theta_s}^{L},
$$

where $B=Q^{\top}K$ selects sources and $C=OV$ transports content. Existing work
contains strong results on special training problems and on prescribed-kernel depth
dynamics. Within the primary sources and their reference chains reviewed here, no
single theorem closes this full map for a factorized, identifiable exact-softmax task.
This is a scoped literature statement, not a logical claim about every publication.

The target is conditional, not universal. The missing result is a theorem that starts
from an identifiable structured task, finite representability, and factor access, then
derives both a task-weighted kernel error and the constants needed to propagate it
through depth. Assuming an already aligned attention map would leave the training gap
untouched.

## Evidence basis for candidate conditions

The condition list is an evidence ledger, not a menu of convenient assumptions:

| Candidate | Extracted from | Supporting evidence | What is not yet known |
|---|---|---|---|
| task identifiability | the repeated-key edge in MQAR and the two-parent LEGO transition | the published task laws make the required source intervention explicit; the finite support makes its matched effect exactly enumerable | the weakest identifiability condition for general structured tasks |
| representability | the exact bounds $\operatorname{rank}(B_h),\operatorname{rank}(C_h)\le d_h$ | the ICML 2020 low-rank theorem establishes the architectural bottleneck; the paired dense and rank-matched controls distinguish a full-rank capacity upper bound from a coordinate control | a sharp task-dependent rank or approximation threshold |
| factor access | the exact composite preconditioners induced by $B=Q^{\top}K$ and $C=OV$ | matrix-factorization theory motivates balancing; the repository proves the $Q=K=0$ trap and a positive role-tied MQAR trajectory | a minimal matrix-valued accessibility condition |
| depth stability | the exact target update for the published LEGO recurrence | fixed-kernel Transformer dynamics and the LEGO theorem supply comparison objects | constants must be derived for the learned exact-softmax model; they are not a fourth assumption |

A candidate becomes a theorem hypothesis only if its mathematical necessity or
sufficiency is established in the specified task/model class. Toy and Pythia
measurements can reject or motivate a candidate, but cannot create one by themselves.

## Fixed-kernel depth dynamics

[A Mathematical Perspective on
Transformers](https://arxiv.org/abs/2312.10794) treats tokens as interacting particles
and studies clustering and measure dynamics under prescribed interaction laws. Section
10 of the current arXiv version identifies parameter training as outside its analysis.
The [Transformer PDE](https://arxiv.org/abs/2501.18322) develops well-posedness and
mean-field limits for related masked and multi-head depth dynamics.

These works study equations of the form

$$
\partial_t\mu_t
+
\nabla\cdot\left(
\mu_t\,\mathcal V_{B,C}[\mu_t]
\right)
=0
$$

with $B,C$ given. Our training variable $s$ must remain distinct from the depth
variable $t$.

[Training-Induced Escape from Token
Clustering](https://arxiv.org/abs/2605.07772) connects training to a noisy mean-field
model, but trains a parameter-linear FFN under prescribed attention. It does not learn
$E,Q/K,O/V$ jointly and therefore does not close the target map.

## Training results that are prior art

| Primary source | Established result | Boundary relative to this project |
|---|---|---|
| [Max-Margin Token Selection, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html) | gradient descent and regularization paths select max-margin tokens in simplified attention | not joint learned representations, QK/OV transport, and depth dynamics |
| [Scan and Snap](https://arxiv.org/abs/2305.16380) | token selection emerges from co-occurrence in a one-layer next-token model | no general learned matrix-valued kernel |
| [Co-occurrence via Gradient Flow, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/520416e27d3b0cef3cd70a083e2991c7-Abstract-Conference.html) | two-stage gradient-flow dynamics for jointly trained attention matrices and a linear MLP | one layer and specialized orthogonal classification data |
| [Unveiling Induction Heads](https://arxiv.org/abs/2409.10559) | a two-layer multi-head softmax model learns an induction circuit on Markov data | specialized copier-selector-classifier structure |
| [Multi-head softmax ICL](https://openreview.net/forum?id=3TM3fxwTps) | learned QK/OV head structure for Gaussian linear regression | no learned token dictionary or general interaction graph |
| [Provably Learn CoT, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b86a195e70f27017c514fa0e5f80595f-Abstract-Conference.html) | gradient descent learns attention concentration for LEGO state tracking and proves length generalization | the LEGO learnability claim is already solved; our increment must connect factorized kernel learning to depth error |
| [Infinite Limits of Multi-head Transformer Dynamics, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3eff068e195daace49955348de9f8398-Abstract-Conference.html) | DMFT for infinite-head, key-width, and depth limits | statistical limits do not identify a finite task-required graph |

Consequently, the following are not contributions: attention can be trained to select
tokens; low-rank attention has an expressivity bottleneck; QK/OV can align on a special
task; prescribed QKV can cluster tokens.

## Mathematical tools

| Tool | Relevant source | Use here | Limitation |
|---|---|---|---|
| interacting particles and Vlasov transport | [Perspective](https://arxiv.org/abs/2312.10794), [Transformer PDE](https://arxiv.org/abs/2501.18322) | propagate a learned-kernel error through depth | does not select the kernel |
| Wasserstein neural mean field | [Mei, Misiakiewicz, Montanari, COLT 2019](https://proceedings.mlr.press/v99/mei19a.html) | parameter-distribution limits and finite-width control | multiplicative softmax factors require a new closure |
| matrix-factor balancing | [Du, Hu, Lee 2018](https://arxiv.org/abs/1806.00900), [Arora et al. 2019](https://arxiv.org/abs/1905.13655) | Gram invariants, gauge, and composite preconditioning | attention is not linear matrix sensing |
| max-margin implicit bias | [Max-Margin Token Selection](https://proceedings.neurips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html) | candidate proof method for diverging score margin | does not control value transport or residual depth |
| associative-memory energy | [Hopfield Networks Is All You Need](https://openreview.net/forum?id=tL89RnzIiCd) | fixed-point and retrieval interpretation of softmax | fixed memories are not training-induced kernels |
| low-rank expressivity | [Low-Rank Bottleneck in Multi-head Attention, ICML 2020](https://proceedings.mlr.press/v119/bhojanapalli20a.html) | known capacity boundary for $d_h$ | rank itself is not new |

The proof strategy is therefore narrow: use task symmetry and factor balancing to close
order parameters; use gradient-flow or max-margin arguments for score/value alignment;
then use stability or transport bounds for finite depth.

## Structured data

| Data | Primary source | Known task graph | Role |
|---|---|---|---|
| MQAR | [Zoology, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html), [official code](https://github.com/HazyResearch/zoology) | repeated key to its associated value | one-step kernel-learning theorem |
| LEGO | [Provably Learn CoT, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b86a195e70f27017c514fa0e5f80595f-Abstract-Conference.html) | predicate and previous state to next state | multi-step training-to-depth theorem |
| CLRS | [CLRS, ICML 2022](https://proceedings.mlr.press/v162/velickovic22a.html) | algorithm hints | external validation only after LEGO |
| Pythia | [Pythia](https://arxiv.org/abs/2304.01373) | no unique internal graph | instrumentation calibration only |
| PolyPythias | [PolyPythias](https://arxiv.org/abs/2503.09543) | no unique internal graph, but multiple seeds | later external trajectory replication |

The official Zoology MQAR generator samples distinct keys and values, interleaves
key-value pairs, places repeated key queries later, and labels each query with its
associated value. The theorem uses its single-query binary-value specialization and
states that restriction explicitly.

The published LEGO distribution samples variables without replacement, an initial
state uniformly, and actions independently with replacement, then applies the actions
recursively. The active code implements the cyclic simply-transitive specialization.

## Current research map

| Category | Exact status |
|---|---|
| Solved | prescribed-kernel clustering/PDE; several special training-selection theorems; low-rank expressivity; exact $C=3,m=2$ quotient/risk; bounded-quotient impossibility; balanced full-rank wrong branch; four unstable modes at one wrong boundary |
| Existing tools requiring new verification | max-margin direction, matrix-factor balancing, equivariant mode decomposition, center-stable-manifold avoidance, stochastic approximation |
| Experimental fact without general theory | dense composite controls reduce toy leakage while rank-matched controls do not; Pythia routing is nonmonotone and template-dependent |
| Current theorem target | almost-everywhere correct-boundary selection for factorized matrix MQAR; only afterward, learned-kernel error propagation through LEGO depth |

The new MQAR results prove the positive radial statement and refute both bounded
quotient dynamics and unrestricted balanced initialization. The immediate gap is now
precise: compactify the diverging score margin, classify all factor-access singular
families, determine their center-stable sets, and exclude unclassified escape for
almost every natural initialization. Additional model scale does not close this gap.
