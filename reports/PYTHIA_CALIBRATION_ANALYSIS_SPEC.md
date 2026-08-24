# Pythia-70M Training-Trajectory Calibration Specification

Frozen before the float64-v4 run. The completed run uses the same design and thresholds.

## 1. Scope

The study asks whether the toy measurements can be executed without ambiguity on a
real pretrained exact-softmax Transformer. It does not estimate a population training
law.

The eight checkpoints belong to one Pythia-70M-deduped trajectory. Checkpoints,
templates, prompts, layers, heads, and memory slots are repeated measurements, not
independent seeds. The study reports deterministic finite-population descriptions only;
it reports no seed-level confidence interval or p-value.

## 2. Frozen trajectory and prompt population

Repository: EleutherAI/pythia-70m-deduped.

Revisions:

$$
\{\text{step0},64,512,1000,4000,16000,64000,143000\}.
\tag{PC1}
$$

Each revision evaluates four frozen templates. Each template has 16 skeletons and the
complete $2^4=16$ binary-value cube, giving 256 prompts per template and 1024 prompts
per revision.

Each skeleton contains four distinct concept-value memory cards and one target query.
The memory values are the strings plus and minus. The teacher-forced answer suffixes
are separate strings with their required leading token boundary. This prevents the
double-space and prompt/answer tokenization confounds found during pre-production
review.

The complete prompt-plus-answer string is tokenized jointly. The separately tokenized
prompt must be an exact prefix. Fast-tokenizer offsets must prove that a counterfactual
changes only the registered concept span and that all value-bearing memory-card spans
are pairwise disjoint.

All revisions share one prompt-population hash, tokenizer contract, measurement-contract
hash, source hash bundle, and execution-environment identity. Resume refuses a changed
environment, backend flags, source, or population.

## 3. Output and functional decomposition

Let $\ell_+$ and $\ell_-$ be complete-answer conditional log likelihoods. The bounded
scalar output is

$$
f_\theta(X)
=
\frac{e^{\ell_+}-e^{\ell_-}}
{e^{\ell_+}+e^{\ell_-}}
=
\tanh\left(\frac{\ell_+-\ell_-}{2}\right).
\tag{PC2}
$$

For label $Y\in\{-1,+1\}$,

$$
R=\frac12\mathbb E(f-Y)^2,
\qquad
A=\mathbb E\,\mathbf 1\{\operatorname{sign}(f)=Y\}.
\tag{PC3}
$$

For a fixed skeleton and value cube,

$$
\widehat f_S
=
2^{-4}\sum_{v\in\{-1,+1\}^4}
f(v)\prod_{i\in S}v_i.
\tag{PC4}
$$

The stored decomposition is

$$
E_T=(\widehat f_{\{J\}}-1)^2,
\qquad
L_D=\sum_{i\ne J}\widehat f_{\{i\}}^2,
\tag{PC5}
$$

$$
L_H=\sum_{|S|\ge2}\widehat f_S^2,
\qquad
L_0=\widehat f_\varnothing^2,
\qquad
L_W=L_D+L_H+L_0,
\tag{PC6}
$$

with exact audit

$$
2R=E_T+L_W.
\tag{PC7}
$$

Target-value flipping is computed independently and must agree with
$\widehat f_{\{J\}}$.

## 4. Distinct interventions

### Natural on-support swap

Replace one distractor concept by an absent, token-length-matched concept while
preserving positions, values, target, and label:

$$
I_{\rm swap}=\mathbb E\left[f(X')-f(X)\right]^2.
\tag{PC8}
$$

This is a function-level robustness effect. The calibration does not retain the raw
episode-level swap distribution, so it cannot establish a sparse-collision tail law.

### Registered direct-edge effect

For memory card $i$, set the final prompt decision position's attention score to every
token in that full value-bearing card span to $-\infty$ in every layer and head. Then
recompute the complete answer score:

$$
\delta_i
=
Y\left[f(X)-f(X^{(-i)})\right].
\tag{PC9}
$$

The registered statistic is

$$
S_{\rm key}
=
\mathbb E\left[
\delta_J
-
\frac13\sum_{i\ne J}\delta_i
\right].
\tag{PC10}
$$

Raw episode-by-slot effects are stored in numeric NPZ and reconstructed by the strict
reader. This intervention measures one direct receiver-to-card path. It is not total
mediation because later answer tokens and indirect token paths remain.

### Finite activation patches

Three roles remain separate:

1. **source-span transmission:** replace the changed memory concept span;
2. **decision-receiver state:** replace the final prompt decision position;
3. **coherent replay:** replace every differing activation at a registered residual
   site.

Each patch uses the donor activation and reruns the actual nonlinear suffix. The roles
overlap and are not an additive QK/OV/FFN decomposition.

## 5. Head and residual diagnostics

Observation-only hooks store post-RoPE Q/K, V, reconstructed attention probabilities,
and per-head pre-OV mixtures. They must leave logits, weights, gradients, hooks, model
mode, and RNG unchanged.

For each layer and head, report target-card attention, mean distractor-card attention,
and their difference. Attention mass is descriptive and cannot replace (PC10).

Pythia uses parallel residual blocks. For matched base and donor executions,

$$
\Delta h_{\rm post}
=
\Delta h
+
\Delta h_{\rm attn}
+
\Delta h_{\rm ffn}.
\tag{PC11}
$$

The calibration stores all branch chords and the closure residual. Sequential-residual
FFN formulas are not applied.

## 6. Completion gates

The run is complete only if all conditions hold:

1. all 8 revision directories and the root have atomic success markers;
2. the failure ledger has no unresolved revision;
3. there are exactly 32 revision-template rows;
4. each revision has exactly 4096 direct-edge episode-slot rows, 196608
   episode-layer-head-slot observation rows, 24576 episode-layer-site patch rows, and
   6144 episode-layer parallel-residual rows;
5. every revision shares one execution-environment identity, including deterministic,
   TF32, and cuDNN flags;
6. schema, contract, source, tokenizer, and prompt hashes agree;
7. raw NPZ, aggregate JSON/CSV, revision rows, and root aggregates reconstruct exactly;
8. the maximum absolute residual in (PC11) is at most $10^{-5}$;
9. observation-only measurements have no model or RNG side effect.

A failed checkpoint cannot be removed from the trajectory. Instrumentation must be
fixed prospectively in a new output directory and schema.

## 7. Frozen figures

All figures use revision on the horizontal axis and display all four templates:

1. accuracy, risk, and value-flip effect;
2. $E_T,L_D,L_H,L_0,L_W$;
3. natural swap, target edge, mean distractor edge, and $S_{\rm key}$;
4. layer-head target/distractor routing;
5. three separately faceted patch roles;
6. parallel-residual chord norms and closure;
7. template heterogeneity.

Every caption states: one pretraining trajectory; checkpoint is a repeated measure;
calibration only.

## 8. Completed float64-v4 outcome

The strict reader accepts 8/8 revisions and 32 checkpoint-template cells. The maximum
parallel-residual closure error is

$$
2.043\times10^{-14}.
$$

Four-template mean accuracy and $S_{\rm key}$ peak at step 16000 and decline by the
final checkpoint:

$$
A_{16000}=0.6299,
\qquad
A_{143000}=0.5430,
$$

$$
S_{\rm key,16000}=0.1040,
\qquad
S_{\rm key,143000}=0.0299.
$$

Mean best-head attention selectivity rises over the same interval, so attention-mass
selectivity is not equivalent to the direct-edge effect. The proposed universal
diffuse-to-selective-to-sparse-collision/downstream-reorganization narrative is not
supported. The calibration supplies validated measurement code, not a population
training conclusion.
