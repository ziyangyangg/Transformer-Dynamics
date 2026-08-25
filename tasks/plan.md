# Active plan: classify single-layer kernel-selection basins

## Fixed objective

For the complete $C=d=3,m=2$ matrix-MQAR population, write
$\theta_0=\varepsilon\xi$ with $\xi\in\mathbb S^{41}$. Construct a checkable
success region $\mathcal G$ and explicit failure regions $\mathcal F_j$ so that,
up to a null remainder, gradient flow either learns

$$
\kappa^*=((1,0))_{q\ne d}
$$

or enters one certified wrong boundary. The region definitions must follow from the
data signal and factor dynamics. They may not assume aligned attention, $K=Q$, a
positive gain, exact balance, or a uniform pullback constant.

This remains a condition-discovery theorem: the conditions are the output of the
classification, not premises chosen to force convergence. A pullback lower bound may
be used only as one sufficient certificate; it cannot replace basin derivation.

## Completed

- Exact population risk, raw gradients, balance tensors, and kernel-access identity.
- Positive convergence theorem on the role-tied positive branch.
- Finite quotient critical-point and raw access-singularity analysis.
- Verified open large-norm wrong basin with $R\to1/2$.
- Verified positive-density small-initialization basin with $R\to1/4$.
- M1 standard-Transformer boundary study: 60 RTX 5090 trajectories, 20 paired seeds,
  exact-zero and $2^{-8}$ Q/K interventions.

## Next proof gates

1. Classify contrast-orientation flow at the score origin.
2. Prove retained contrast access on a candidate success region.
3. Search its complement for further scale-uniform open failure basins.
4. Prove exhaustion: every remaining trajectory learns $\kappa^*$ or enters a listed
   failure basin.
5. Only then lift the classification to general finite structured-selection
   populations, followed by LEGO one-step routing and depth composition.

## Stop rules

- M1 uses AdamW and cannot certify continuous-time gradient flow.
- Accuracy, attention mass, and full-card blocking are not interchangeable with a
  learned selective score kernel.
- Do not add another model family or dataset before the basin classification closes.
- Any open wrong basin is retained as a necessary-condition witness, not discarded as
  an optimization failure.
