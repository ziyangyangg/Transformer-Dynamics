# Fixed-Parameter Perspective Clustering Baseline

## Result

The prescribed-kernel sphere dynamics collapse 64 particles in dimension 3 to one
direction. At the endpoint, however, attention is uniform. This is a fixed-parameter
depth-dynamics baseline; it is not selective retrieval and it does not explain how
training learns a task-aligned kernel.

## Model

The implementation reproduces the public sphere experiment associated with
*A Mathematical Perspective on Transformers*. Let $z_i(t)\in\mathbb S^{d-1}$ and

$$
a_{ij}(t)
=
\frac{
\exp\left(\beta z_i(t)^{\top}z_j(t)\right)
}{
\sum_{k=1}^{n}
\exp\left(\beta z_i(t)^{\top}z_k(t)\right)
}.
\tag{C1}
$$

With $A=V=I$, one Euler step is

$$
\widetilde z_i(t+\Delta t)
=
z_i(t)
+
\Delta t\sum_{j=1}^{n}a_{ij}(t)z_j(t),
\tag{C2}
$$

$$
z_i(t+\Delta t)
=
\frac{\widetilde z_i(t+\Delta t)}
{\|\widetilde z_i(t+\Delta t)\|_2}.
\tag{C3}
$$

The registered configuration is

$$
n=64,
\qquad
d=3,
\qquad
\beta=1,
\qquad
\Delta t=0.1,
\qquad
T=15,
$$

with seed 20260815 and the legacy NumPy random stream used by the source experiment.

## Measurements

Let $G=ZZ^{\top}$ be the Gram matrix. The mean off-diagonal cosine is

$$
\rho
=
\frac{\mathbf 1^{\top}G\mathbf 1-n}{n(n-1)}.
\tag{C4}
$$

The Gram participation rank is

$$
r_G
=
\frac{(\operatorname{tr}G)^2}
{\operatorname{tr}(G^2)}.
\tag{C5}
$$

The row-wise attention entropy is

$$
H_i=-\sum_j a_{ij}\log a_{ij}.
\tag{C6}
$$

The implementation also records the minimum, median, and 90th percentile pairwise
cosines, the mean displacement, the top Gram eigenvalues, and exact identity residuals.

## Numerical result

| Quantity | Initial | Final |
|---|---:|---:|
| mean off-diagonal cosine $\rho$ | 0.01447 | 0.9999999996 |
| Gram participation rank $r_G$ | 2.7948 | 1.0000000007 |
| mean attention entropy | nonuniform | $\log 64$ |
| mean attention weight | nonuniform | $1/64$ |

Threshold times:

| Event | First time |
|---|---:|
| $\rho\ge0.5$ | 3.3 |
| $\rho\ge0.9$ | 4.7 |
| $r_G\le1.1$ | 5.2 |
| at least 90% of pairs have cosine at least 0.9 | 5.3 |

The final attention matrix is uniform because all particle directions are equal:

$$
a_{ij}(T)=\frac1n.
\tag{C7}
$$

Thus global clustering does not imply a task-specific source choice.

## Reproduction

```bash
PYTHONPATH=src python -m routing_lab.clustering_baseline \
  --output-directory results/clustering-baseline-v1
PYTHONPATH=src python -m unittest -v tests.test_clustering_baseline
```

The result directory contains deterministic JSON/CSV trajectories, initial/final Gram
matrices, and PNG/SVG figures. Tests verify the update equation, random stream, Gram
identities, and byte stability.

## Scientific position

This baseline supplies the right-hand side of the research program: given a fixed
interaction law, it shows one possible depth trajectory. It does not supply the
left-hand side:

$$
(\mathcal D,R,\theta_0)
\longrightarrow
B_s,C_s.
$$

It therefore cannot be cited as evidence that gradient training learned the correct
routing kernel. Its main negative control is exact: complete geometric collapse can
coexist with maximally diffuse attention.
