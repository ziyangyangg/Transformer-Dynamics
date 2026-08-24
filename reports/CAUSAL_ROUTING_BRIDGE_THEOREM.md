# From Low Risk to Direct-Edge Routing

## Scope

This report concerns a one-layer, value-linear exact-softmax subclass. It does not
state a result for a general residual Transformer with RMSNorm, FFNs, or indirect
multi-layer paths.

## Model and intervention

Fix an episode skeleton $\omega=(c_{1:m},J)$ with $m\ge2$. Values are independent
Rademacher variables and $Y=v_J$. For head $h$, let $a_{hi}>0$ be the memory-slot
weights and $a_{h0}>0$ the query-self weight:

$$
a_{h0}+\sum_{i=1}^{m}a_{hi}=1.
\tag{B1}
$$

The weights may depend on the keys, target, and positions, but not on the values.
Query-self is zero in the measured value channel. There is no memory relay, FFN,
additional value bypass, or output nonlinearity. The model is

$$
f_\omega(v)
=
b_\omega+
\sum_{h=1}^{H}g_{h\omega}
\sum_{i=1}^{m}a_{hi\omega}v_i,
qquad
g_{h\omega}\ge0.
\tag{B2}
$$

Blocking slot $i$ removes its score edge and renormalizes every remaining softmax
weight:

$$
a_{hk}^{(-i)}=\frac{a_{hk}}{1-a_{hi}}
\quad(k\ne i),
qquad
a_{hi}^{(-i)}=0.
\tag{B3}
$$

Define

$$
\delta_i
=
v_J\left[f_\omega(v)-f_\omega^{(-i)}(v)\right],
\tag{B4}
$$

$$
s_{\rm key}(\omega)
=
\mathbb E_v\left[
\delta_J-
\frac1{m-1}\sum_{i\ne J}\delta_i
\right],
\qquad
S_{\rm key}=\mathbb E_\omega s_{\rm key}(\omega),
\tag{B5}
$$

and

$$
R
=
\frac12\mathbb E_{\omega,v}
\left(f_\omega(v)-v_J\right)^2.
\tag{B6}
$$

## Theorem

**Theorem B1.** Under (B1)-(B6),

$$
S_{\rm key}\ge1-\sqrt{2R}.
\tag{B7}
$$

**Proof.** The target value coefficient is

$$
\kappa_J(\omega)=\sum_h g_h a_{hJ}.
\tag{B8}
$$

Rademacher orthogonality and the renormalization in (B3) give

$$
\mathbb E_v\delta_J
=
\sum_hg_ha_{hJ}
=
\kappa_J,
\tag{B9}
$$

and, for every $i\ne J$,

$$
\mathbb E_v\delta_i
=
-\sum_h
g_h\frac{a_{hi}a_{hJ}}{1-a_{hi}}
\le0.
\tag{B10}
$$

Hence $s_{\rm key}(\omega)\ge\kappa_J(\omega)$. Parseval's identity gives

$$
\mathbb E_\omega(\kappa_J-1)^2\le2R.
\tag{B11}
$$

Cauchy-Schwarz then yields

$$
S_{\rm key}
\ge\mathbb E\kappa_J
\ge1-\mathbb E|\kappa_J-1|
\ge1-\sqrt{2R}.
$$

This proves (B7). The theorem constrains a blocking effect, not attention mass.
A large score margin additionally requires an upper bound on the effective gain.

## Exact signed-gain counterexample

Remove only the condition $g_h\ge0$. Let $m=2$ and $H=2$. For each skeleton, order
the positions as target, distractor, self and set

$$
a_1=
\left(\frac15,\frac3{10},\frac12\right),
\qquad
a_2=
\left(\frac8{35},\frac25,\frac{13}{35}\right),
\tag{B12}
$$

$$
g_1=35,
\qquad
g_2=-\frac{105}{4},
\qquad
b=0.
\tag{B13}
$$

All attention probabilities are strictly positive and sum to one. Logits equal to
their logarithms produce these probabilities exactly.

The target and distractor function coefficients are

$$
35\frac15-\frac{105}{4}\frac8{35}=1,
\tag{B14}
$$

$$
35\frac3{10}-\frac{105}{4}\frac25=0.
\tag{B15}
$$

Therefore $f(v)=v_J$ for all four value assignments, so $R=0$. Nevertheless,

$$
\mathbb E_v\delta_J=1,
\tag{B16}
$$

while (B10), without its sign conclusion, gives

$$
\mathbb E_v\delta_D
=
-35\frac{(3/10)(1/5)}{1-3/10}
+
\frac{105}{4}
\frac{(2/5)(8/35)}{1-2/5}
=
-3+4
=
1.
\tag{B17}
$$

Hence

$$
S_{\rm key}=1-1=0.
\tag{B18}
$$

This is an exact-softmax counterexample: two heads compute the correct function, but
signed OV/readout cancellation makes target and distractor blocking effects identical.

## Composite score realization

Let $e_c$ be orthonormal concept vectors and let $q_0,m_0$ be orthogonal query and
memory type vectors. Write the three desired head logits as
$\ell_{hT},\ell_{hD},\ell_{hS}$. The composite

$$
B_h
=
(\ell_{hT}-\ell_{hD})
\sum_c e_ce_c^{\top}
+
\ell_{hD}q_0m_0^{\top}
+
\left[
\ell_{hS}-(\ell_{hT}-\ell_{hD})
\right]q_0q_0^{\top}
\tag{B19}
$$

assigns exactly these logits to matching memory, nonmatching memory, and query-self.
The value direction is orthogonal to these basis vectors and lies in the null space of
$B_h$, so scores are value-blind. Temperature and the factor $1/\sqrt{d_h}$ can be
absorbed into $B_h$.

## Boundary

Established:

$$
\text{value-blind scores}
+
\text{nonnegative gains}
+
\text{no bypass}
\Longrightarrow
S_{\rm key}\ge1-\sqrt{2R}.
$$

Not established: the same implication for signed gains, indirect paths, RMSNorm, FFNs,
or a general multi-layer Transformer. The counterexample proves that any broader theorem
must bound signed cancellation or include it explicitly in the error budget.
