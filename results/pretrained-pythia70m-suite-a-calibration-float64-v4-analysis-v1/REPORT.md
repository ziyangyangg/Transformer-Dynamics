# Pythia-70M Float64 Calibration: Stage Decision

Audit: all 8 checkpoints and all 32 checkpoint-template rows are complete; raw P10 slot interventions reconstruct P11 exactly. The maximum parallel-residual closure error is `2.043e-14` against the registered `1e-5` threshold. The statistical unit is **one pretraining trajectory**.

Definitions: $f=\tanh((\log p(\text{plus})-\log p(\text{minus}))/2)$, $R=\mathbb{E}[(f-Y)^2]/2$, and $S_{\mathrm{key}}=\mathbb{E}[\delta_J-\operatorname{mean}_{i\ne J}\delta_i]$, where $\delta_i=Y(f-f^{(-i)})$ blocks only the direct edge from the final prompt receiver to memory card $i$.

Observations: the maximum accuracy is `0.715` (step4000/bracket_dictionary) and the minimum risk is `0.379`. $S_{\mathrm{key}}$ ranges from `-5.878e-05` to `2.230e-01`. The largest observation-only head selectivity is `1.840e-01` (step143000/line_records/L5H4); the maximum natural-swap MSE is `3.578e-02`. Final checkpoint: compact_cards: acc=0.535, R=0.510, L_W=0.163, S_key=0.0402; line_records: acc=0.531, R=0.507, L_W=0.147, S_key=0.0226; prose_facts: acc=0.543, R=0.503, L_W=0.074, S_key=-5.88e-05; bracket_dictionary: acc=0.562, R=0.486, L_W=0.136, S_key=0.0569.

Assessment: the proposed `diffuse -> selective routing -> sparse collision -> downstream reorganization` trajectory is **not supported**. The four final templates does not pass the descriptive stable-retrieval screen. Episode-level natural-swap deltas were not stored, so collision sparsity and tail behavior are not testable. The three finite-patch roles are overlapping nonlinear suffix interventions and do not identify QK, OV, or FFN as a unique compensator.

Boundary: checkpoints, templates, layers, and heads are repeated measurements, not independent samples. P10 is not total mediation. These data validate the measurement pipeline and show weak, template-dependent routing signals; they do not establish a general law of GPT training.

Conclusion: `full_story_supported=false`. The theorem-facing target remains the full-matrix MQAR-to-LEGO problem: derive the gradient-flow dynamics of $B=Q^\top K$ and $C=OV$, prove task-kernel alignment under explicit assumptions, and construct counterexamples when those assumptions are removed.
