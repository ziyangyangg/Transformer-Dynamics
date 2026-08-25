# MQAR M1 boundary result

## Decision

The exact `Q=K=0` factor-access boundary is invariant and harmful in this standard
four-layer softmax Transformer. A nonzero initialization only $2^{-8}$ as large as
the standard Q/K scale escapes and learns. This is finite-step AdamW evidence, not a
population-gradient-flow theorem.

## Experiment

Twenty paired training seeds were run for each arm on one NVIDIA GeForce RTX 5090. The model has
four pre-RMSNorm/RoPE attention layers, four heads, $d=128$, FFN width 512, tied embeddings,
and 1,836,160 parameters. Data follow the public Zoology MQAR construction. Training
mixes $(L,m)=(64,4),(128,8),(256,16)$; evaluation also includes $(512,16)$ and
$(1024,32)$. The independent unit is the training seed.

## Observations

Final mean accuracy at $(L,m)=(64,4)$ is 0.9287
(standard), 0.9170 (Q/K scale $2^{-8}$), and
0.0721 (exact zero). At $(256,16)$ the corresponding means are
0.8867, 0.8545, and
0.0069. The exact-zero arm has Q/K factor norm and measured
Q/K gradient norm exactly zero for every seed and checkpoint. The small arm grows from
mean Q/K norm 0.028300 to
21.4935.

The exact-zero arm still reaches 0.0721 accuracy and has mean
full-card blocking contrast 1.5241. Uniform attention can transmit content, so this
blocking statistic is not evidence of learned selective QK routing. Accuracy, edge
importance, and a learned score kernel are distinct quantities.

## Boundary

In this architecture, downstream residual/FFN/value paths do not restore high MQAR
accuracy at the exact bilinear Q/K access singularity. The failure does not automatically
extend to the tested $2^{-8}$ nonzero point under AdamW. This experiment does **not**
prove a continuous-time success-region theorem, identify a unique routing head, classify every
singular boundary, or establish long-context generalization: mean accuracy at
$(1024,32)$ is only 0.4071 (standard) and
0.3835 (small).
