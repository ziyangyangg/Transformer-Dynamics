# MQAR M2 signed-orientation result

## Decision

The standard Transformer repairs or obscures the signed initialization; the reduced invariant branch does not transfer as a standalone condition.

## Experiment

The frozen study uses the public Zoology-compatible MQAR law and the unchanged
four-layer M1 Transformer. Five paired arms, 20 independent
training seeds, and 6400 AdamW steps were run on
NVIDIA GeForce RTX 5090. Layers, heads, checkpoints,
and examples are repeated measurements.

## Observations

At $L=256,m=16$, final mean accuracy is
0.9620 (independent),
0.9569 (positive),
0.9593 (negative),
0.9504 (positive-small), and
0.9602 (negative-small).

The positive-minus-negative simultaneous 95% intervals are
[-0.0114,
0.0067] for standard-scale accuracy and
[-0.0363,
0.0166] for small-scale accuracy. The corresponding
best-head score-margin intervals are
[-15.6631,
5.2947] and
[-11.6016,
8.9298].

The registered labels are standard:
`no_joint_signed_separation` /
`architectural_repair`; small:
`no_joint_signed_separation` /
`architectural_repair`.

The initial factor sign is not conserved: mean $Q/K$ cosine moves from $+1$ to
0.2125 in the positive arm and from
$-1$ to 0.0976 in the negative arm.
On the longest configured evaluation population (L1024_m32), final accuracy is
0.4522 (independent),
0.4581 (positive),
0.4631 (negative),
0.4604 (positive-small), and
0.4448 (negative-small). Therefore M2 does not establish
length extrapolation beyond the training support.

## Theory boundary

$K(0)=\pm Q(0)$ is a controlled factor relation, not the definition of correct
routing. The data-defined target score margin is the routing observable. This
finite-step AdamW experiment can show whether the exact single-layer negative branch
persists or is repaired in the standard architecture. It cannot prove a
gradient-flow basin theorem, necessity of a sign condition, or sufficiency for
kernel learning.
