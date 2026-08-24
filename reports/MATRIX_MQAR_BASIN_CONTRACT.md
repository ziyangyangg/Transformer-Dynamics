# Matrix MQAR: Basin-Theorem Contract

## Question

For the complete $C=3,m=2$ MQAR population and factorized gradient flow,
determine whether balanced, full-rank initialization forces selection of the
task-aligned retrieval boundary.

The identifiable coordinates are

$$
\Psi(\theta)=(S,g),\qquad S=E Q^\top K E^\top,\qquad
g=w^\top OVu.
$$

## Falsifiable claims

1. **Bounded-quotient claim.** Can $R\to0$ while $(S,g)$ remains bounded?
2. **Balanced-access claim.** Do full rank and
   $QQ^\top=KK^\top$ exclude a wrong retrieval boundary?
3. **Positive-orientation lemma.** On $K=Q$, what pointwise pullback bound is
   available, and which uniform singular-value bound is still missing?

## Acceptance criteria

- Prove or refute Claims 1 and 2 exactly, without relying on numerical search.
- Encode both statements as float64 regression tests against the complete
  population oracle.
- Audit one positive-orientation and one negative-orientation trajectory with
  two-tolerance DOP853; numerical results are verification only.
- State the corrected convergence target.  Do not invoke compact LaSalle on an
  unbounded quotient, and do not promote balance to an orientation condition.
