# Matrix MQAR boundary selection: a positive-measure wrong basin

The proposed almost-everywhere convergence theorem is false.  The obstruction is
an ambient open horn at large score-factor norm.  In this horn the value gain is
negative, the score dynamics maximizes a wrong homogeneous margin, every attention
weight tends to zero, and the population risk tends to \(1/2\).

All arguments below are finite-dimensional and self-contained.  No external
theorem is invoked.

# lemma lem:risk-identity

## statement

For every ordered pair \(i\ne j\), abbreviate
\[
a_{ij}=a_{i\mid ij},\qquad b_{ij}=b_{j\mid ij}.
\]
Then
\[
R(\theta)
=\frac1{12}\sum_{i\ne j}
\left[(g a_{ij}-1)^2+(g b_{ij})^2\right].
\tag{1}
\]
Consequently, if \(g a_{ij}\to0\) and \(g b_{ij}\to0\) for all six
ordered pairs, then \(R\to1/2\) and the target conclusion in Theorem T
fails.

## proof

Condition on the ordered target--distractor pair \((i,j)\).  The two values
\(v_i,v_j\) are independent Rademacher variables and
\[
f_\theta-Y=(g a_{ij}-1)v_i+g b_{ij}v_j.
\]
The mixed term has zero conditional expectation, so
\[
\mathbb E_{v_i,v_j}\bigl[(f_\theta-Y)^2\bigr]
=(g a_{ij}-1)^2+(g b_{ij})^2.
\]
There are six equally likely ordered pairs and the definition of the risk has
the prefactor \(1/2\).  This proves (1).  If both delivered coefficients tend
to zero, each summand in (1) tends to one, whence \(R\to6/12=1/2\);
also \(\lvert g a_{ij}-1\rvert\to1\). \(\square\)

# lemma lem:sharp-wrong-margin

## statement

Put
\[
D=-K,\qquad \Phi=(E,Q,D),\qquad
\|\Phi\|^2=\|E\|_F^2+\|Q\|_F^2+\|D\|_F^2,
\]
and define the degree-four matrix
\[
H(\Phi)=EQ^\top DE^\top=-S.
\]
On the unit sphere \(\|\Phi\|=1\), let
\[
q_i(\Phi)=H_{ii}(\Phi),\qquad
p_{ij}(\Phi)=H_{ij}(\Phi),\qquad
\Gamma(\Phi)=\min_i q_i(\Phi).
\]
Then
\[
\max_{\|\Phi\|=1}\Gamma(\Phi)=\frac1{24}.
\tag{2}
\]

Moreover, equality in (2) is possible only when, for some unit vectors
\(v,\ell\in\mathbb R^3\) and some
\(\sigma=(\sigma_1,\sigma_2,\sigma_3)\in\{-1,+1\}^3\),
\[
E=\frac{\widehat\sigma v^\top}{\sqrt2},\qquad
Q=D=\frac{\ell v^\top}{2},\qquad
\widehat\sigma=\frac{\sigma}{\sqrt3},
\tag{3}
\]
up to the redundant simultaneous sign choices in this representation.  At
such a point
\[
H_{ij}=\frac{\sigma_i\sigma_j}{24}.
\tag{4}
\]
In particular, the compact component
\[
\mathcal M_+
=\left\{
\left(\frac{p v^\top}{\sqrt2},
      \frac{\ell v^\top}{2},
      \frac{\ell v^\top}{2}\right):
v,\ell\in\mathbb S^2
\right\},
\qquad p=\frac{\mathbf1}{\sqrt3},
\tag{5}
\]
consists of global maximizers for which every entry of \(H\) equals
\(1/24\).

There is an open spherical neighborhood \(U\) of \(\mathcal M_+\), with
compact closure, and constants \(\mu,\delta,\Delta>0\) such that
\[
q_i(u)\ge\mu,\qquad p_{ij}(u)\ge\mu,
\tag{6}
\]
\[
2p_{ij}(u)-q_k(u)\ge\delta
\quad(i\ne j,\ k\in\{1,2,3\})
\tag{7}
\]
for \(u\in\overline U\), while
\[
\Gamma(u)\le\frac1{24}-4\Delta
\quad\text{for }u\in\partial U.
\tag{8}
\]

## proof

Let \(z_i=E^\top e_i\).  Then
\[
q_i=(Qz_i)^\top(Dz_i)
\le \|Qz_i\|\,\|Dz_i\|
\le \|Q\|_F\|D\|_F\|z_i\|^2.
\tag{9}
\]
Write
\[
x=\|E\|_F^2,\qquad y=\|Q\|_F^2,\qquad z=\|D\|_F^2.
\]
On the unit sphere \(x+y+z=1\).  Summing (9) and using the minimum,
the arithmetic--geometric mean inequality, and then a one-variable
maximization gives
\[
3\Gamma
\le x\sqrt{yz}
\le \frac{x(1-x)}2
\le\frac18.
\tag{10}
\]
Thus \(\Gamma\le1/24\).

Let \(P_0=\mathbf1\mathbf1^\top/3=pp^\top\).  The tuple
\[
\Phi_*=\left(\frac{P_0}{\sqrt2},\frac{P_0}{2},\frac{P_0}{2}\right)
\]
has norm one, and
\[
H(\Phi_*)=\frac18P_0.
\]
Every entry is \(1/24\), so the upper bound is sharp.

We record the equality case because it is needed to isolate the positive
component.  Equality in (10) forces
\[
x=\frac12,\qquad y=z=\frac14,
\tag{11}
\]
and equality in every summand of (9).  In particular all \(q_i=1/24\)
and all three row vectors \(z_i\) have squared norm \(1/6\).
For a nonzero vector \(z_i\), equality in
\[
\|Qz_i\|\le\|Q\|_{\mathrm{op}}\|z_i\|
\le\|Q\|_F\|z_i\|
\]
forces \(Q\) to have rank one and \(z_i\) to lie in its right singular
line.  The same holds for \(D\).  Equality in the first
Cauchy--Schwarz inequality in (9), together with \(q_i>0\), aligns the
two left singular directions positively.  Hence there are unit vectors
\(v,\ell\) such that
\[
Q=D=\frac{\ell v^\top}{2}.
\]
All rows of \(E\) are multiples of \(v^\top\), and their equal squared
norms give
\[
E=\frac{\widehat\sigma v^\top}{\sqrt2}
\]
for a sign vector \(\sigma\).  Direct multiplication proves (4).
Conversely, every tuple (3) is an equality case.  This proves the
classification and (5).

At every point of \(\mathcal M_+\), all the quantities in (6) equal
\(1/24\), and every left side in (7) equals \(1/24\).  Continuity
therefore gives a neighborhood on which (6)--(7) hold with positive
constants.  By the equality classification, the only global
maximizers in a sufficiently small such neighborhood are the points
of \(\mathcal M_+\).  Choose a smaller metric neighborhood \(U\) whose
closure is still contained in it.  Its boundary is compact and is
disjoint from the global maximizing set.  The continuous function
\(\Gamma\) consequently has maximum strictly below \(1/24\) on
\(\partial U\).  Writing this positive gap as \(4\Delta\) proves
(8). \(\square\)

# lemma lem:exact-tail

## statement

Let \(\Phi=\rho u\), where \(\rho=\|\Phi\|\) and
\(u\in\overline U\).  Suppose that
\[
g=-h,\qquad 0<h_-\le h\le h_+.
\tag{12}
\]
Set
\[
q_i=H_{ii}(\Phi),\qquad p_{ij}=H_{ij}(\Phi),
\]
\[
A_i=e^{-q_i},\qquad B_{ij}=e^{-p_{ij}},\qquad
\mathcal A=\sum_{i=1}^3 A_i.
\tag{13}
\]
Then
\[
a_{ij}=\frac{A_i}{1+A_i+B_{ij}},\qquad
b_{ij}=\frac{B_{ij}}{1+A_i+B_{ij}},
\tag{14}
\]
and the score-factor part of the exact raw gradient flow can be written
\[
\dot\Phi
=\frac h3\sum_i A_i\nabla q_i+\eta
=\frac h3\,\mathcal A\,\nabla\alpha+\eta,
\qquad
\alpha=-\log\mathcal A.
\tag{15}
\]
There are constants \(C,c>0\), uniform for (12),
\(u\in\overline U\), and all sufficiently large \(\rho\), such that
\[
\|\eta\|\le
C\rho^3\mathcal A e^{-c\rho^4}.
\tag{16}
\]

The quotient gain derivative
\(\gamma=\partial_g\mathcal R(S,g)\) satisfies
\[
\gamma=-\frac16\sum_{i\ne j}
\left[a_{ij}+h(a_{ij}^2+b_{ij}^2)\right]<0
\tag{17}
\]
and
\[
|\gamma|\le C\mathcal A.
\tag{18}
\]

## proof

Because \(D=-K\), the change from \(K\) to \(D\) is orthogonal and
the \(\Phi\)-equation remains Euclidean gradient flow.  Expanding (1)
with \(g=-h\) gives the exact identity
\[
R=\frac12+\frac h6\sum_{i\ne j}a_{ij}
  +\frac{h^2}{12}\sum_{i\ne j}(a_{ij}^2+b_{ij}^2).
\tag{19}
\]
Differentiating (14) with respect to \(\Phi\) gives
\[
\nabla a_{ij}
=-a_{ij}(1-a_{ij})\nabla q_i
+a_{ij}b_{ij}\nabla p_{ij},
\tag{20}
\]
\[
\nabla b_{ij}
=a_{ij}b_{ij}\nabla q_i
-b_{ij}(1-b_{ij})\nabla p_{ij}.
\tag{21}
\]
After inserting (20)--(21) into \(\dot\Phi=-\nabla_\Phi R\), the
leading contribution of each ordered pair is
\((h/6)A_i\nabla q_i\).  Each \(i\) occurs with two distractors, so
the sum of these leading terms is the first expression in (15).
For completeness, if \(\eta_{ij}\) denotes the remainder from the
pair \((i,j)\), then
\[
\begin{aligned}
6\eta_{ij}
={}&
\left\{
h[a_{ij}(1-a_{ij})-A_i]
+h^2[a_{ij}^2(1-a_{ij})-a_{ij}b_{ij}^2]
\right\}\nabla q_i\\
&+
\left\{
-h a_{ij}b_{ij}
-h^2a_{ij}^2b_{ij}
+h^2b_{ij}^2(1-b_{ij})
\right\}\nabla p_{ij}.
\end{aligned}
\tag{22}
\]

The elementary bounds
\[
0<a_{ij}\le A_i,\qquad 0<b_{ij}\le B_{ij},\qquad
|a_{ij}-A_i|\le A_i(A_i+B_{ij})
\tag{23}
\]
show that every scalar coefficient in (22) has absolute value at
most
\[
C(A_i^2+A_iB_{ij}+B_{ij}^2).
\tag{24}
\]
By degree-four homogeneity and compactness of \(\overline U\),
\[
\|\nabla q_i(\rho u)\|+\|\nabla p_{ij}(\rho u)\|
\le C\rho^3.
\tag{25}
\]
Conditions (6)--(7) imply
\[
A_i^2\le A_i e^{-\mu\rho^4},\qquad
A_iB_{ij}\le A_i e^{-\mu\rho^4},
\tag{26}
\]
and, for every \(k\),
\[
B_{ij}^2
=e^{-2\rho^4p_{ij}(u)}
\le e^{-\rho^4q_k(u)}e^{-\delta\rho^4}
\le\mathcal A e^{-\delta\rho^4}.
\tag{27}
\]
Summing the finitely many terms in (22), and taking
\(c\le\min\{\mu,\delta\}\), proves (16).
The second equality in (15) follows from
\[
\mathcal A\nabla\alpha
=-\nabla\mathcal A
=\sum_i A_i\nabla q_i.
\]

Finally, differentiating (1) with respect to \(g\) gives
\[
\gamma
=\frac16\sum_{i\ne j}
\left[g(a_{ij}^2+b_{ij}^2)-a_{ij}\right],
\]
which is (17).  The bounds \(a_{ij}\le A_i\), (26), and (27) give
(18). \(\square\)

# lemma lem:soft-margin-barrier

## statement

Under the hypotheses of Lemma \(\mathrm{lem:exact-tail}\), define
\[
m(\Phi)=\frac{\alpha(\Phi)}{\rho^4},
\qquad
\alpha=-\log\sum_i e^{-q_i}.
\tag{28}
\]
For all sufficiently large \(\rho\),
\[
\dot\rho\ge c_1\rho^3\mathcal A>0
\tag{29}
\]
and there is a decreasing function
\[
J(\rho)=C_1\int_\rho^\infty e^{-cs^4}\frac{ds}{s}
\tag{30}
\]
such that
\[
\frac d{dt}\bigl(m(\Phi)-J(\rho)\bigr)\ge0.
\tag{31}
\]
Also, if \(u=\Phi/\rho\), then
\[
\Gamma(u)-\frac{\log3}{\rho^4}
\le m(\Phi)\le\Gamma(u).
\tag{32}
\]

## proof

Introduce the probability vector
\[
\omega_i=\frac{e^{-q_i}}{\mathcal A}
\]
and its Shannon entropy
\[
\mathsf H(\omega)=-\sum_i\omega_i\log\omega_i.
\]
If \(\overline q=\sum_i\omega_iq_i\), then
\[
\alpha=\overline q-\mathsf H(\omega).
\tag{33}
\]
Euler's identity for the degree-four functions \(q_i\) gives
\[
\langle\Phi,\nabla\alpha\rangle
=\sum_i\omega_i\langle\Phi,\nabla q_i\rangle
=4\overline q.
\tag{34}
\]

First ignore the error \(\eta\) in (15).  Along the leading field,
\[
\begin{aligned}
\dot m_{\rm lead}
=\frac h3\mathcal A
\left[
\rho^{-4}\|\nabla\alpha\|^2
-4\alpha\rho^{-6}
 \langle\Phi,\nabla\alpha\rangle
\right].
\end{aligned}
\tag{35}
\]
By Cauchy--Schwarz,
\[
\rho^{-4}\|\nabla\alpha\|^2
\ge\rho^{-6}\langle\Phi,\nabla\alpha\rangle^2.
\]
Combining this with (33)--(34) yields
\[
\dot m_{\rm lead}
\ge
\frac{16h}{3}\mathcal A\rho^{-6}
\overline q\,\mathsf H(\omega)\ge0.
\tag{36}
\]
Here \(\overline q>0\) follows from (6).

On \(\overline U\), homogeneity and compactness give
\[
\|\nabla\alpha\|\le C\rho^3,\qquad
|\alpha|\le C\rho^4,
\]
and hence
\[
\|\nabla m\|\le\frac C\rho.
\tag{37}
\]
Thus the error in (16) can decrease \(m\) by at most
\[
C\rho^2\mathcal A e^{-c\rho^4}.
\tag{38}
\]

Taking the radial component of (15) and using (34) gives
\[
\dot\rho
=\frac{4h}{3\rho}\mathcal A\overline q
+\frac{\langle\Phi,\eta\rangle}{\rho}.
\tag{39}
\]
Condition (6) gives \(\overline q\ge\mu\rho^4\), while (16) makes
the error in (39) exponentially smaller.  Since \(h\ge h_->0\),
(29) follows after increasing the lower threshold for \(\rho\).
Combining (29) and (38) gives
\[
\dot m\ge
-C_1e^{-c\rho^4}\frac{\dot\rho}{\rho}.
\tag{40}
\]
Since
\[
J'(\rho)=-C_1e^{-c\rho^4}/\rho,
\]
equation (40) is exactly (31).

Finally, if \(q_{\min}=\min_i q_i=\rho^4\Gamma(u)\), then
\[
e^{-q_{\min}}\le\sum_i e^{-q_i}
\le3e^{-q_{\min}}.
\]
Apply \(-\log\) and divide by \(\rho^4\) to obtain (32).
\(\square\)

# lemma lem:open-wrong-horn

## statement

There is a nonempty ambient open set
\(\mathcal C\) in the full raw parameter space with the following
properties.

1. \(\mathcal C\cap\Theta_{\rm reg}\) is nonempty and has positive
   Lebesgue measure.
2. Every gradient-flow solution starting in \(\mathcal C\) exists for
   all \(t\ge0\) and remains in \(\mathcal C\).
3. Along every such solution,
   \[
   g(t)<0,\qquad
   \rho(t)\longrightarrow\infty,\qquad
   S_{ij}(t)\longrightarrow-\infty
   \quad\text{for all }i,j.
   \tag{41}
   \]
4. For every ordered pair \(i\ne j\),
   \[
   a_{ij}(t),b_{ij}(t),g(t)a_{ij}(t),g(t)b_{ij}(t)
   \longrightarrow0,
   \qquad R(t)\longrightarrow\frac12.
   \tag{42}
   \]

## proof

Let
\[
\psi=(w,O,V),\qquad \psi_*=(-e_1,I,I).
\]
At \(\psi_*\),
\[
g(\psi_*)=-1,\qquad c_g(\psi_*)=3.
\tag{43}
\]
Choose \(\delta_v>0\) so small that throughout the ball
\(\|\psi-\psi_*\|<\delta_v\),
\[
0<h_-\le -g(\psi)\le h_+,\qquad c_g(\psi)>0,
\tag{44}
\]
and all value-factor norms are bounded by a fixed constant.

The raw value equations are
\[
\dot w=-\gamma OVe_1,\qquad
\dot O=-\gamma w(Ve_1)^\top,\qquad
\dot V=-\gamma O^\top w e_1^\top.
\tag{45}
\]
Equations (18), (44), and (45) imply, while the score direction lies
in \(U\),
\[
\|\dot\psi\|\le C_v\mathcal A
\le C_v'\frac{\dot\rho}{\rho^3},
\tag{46}
\]
where the last inequality is (29).  Choose \(K>C_v'/2\).  The upper
Dini derivative of
\[
\mathcal B(\Phi,\psi)
=\|\psi-\psi_*\|+\frac K{\rho^2}
\tag{47}
\]
satisfies
\[
D^+\mathcal B
\le
C_v'\frac{\dot\rho}{\rho^3}
-2K\frac{\dot\rho}{\rho^3}
\le0.
\tag{48}
\]

Fix \(\rho_0\) sufficiently large that all estimates above hold, that
\[
\frac{\log3}{\rho_0^4}+J(\rho_0)<\Delta,
\qquad
\frac K{\rho_0^2}<\delta_v,
\tag{49}
\]
and define
\[
\mathcal C=
\left\{
\begin{array}{l}
\rho>\rho_0,\quad \Phi/\rho\in U,\\[1mm]
m(\Phi)-J(\rho)>\dfrac1{24}-2\Delta,\\[1mm]
\|\psi-\psi_*\|+\dfrac K{\rho^2}<\delta_v
\end{array}
\right\}.
\tag{50}
\]
Every inequality in (50) is strict and depends continuously on the
raw parameters, so \(\mathcal C\) is ambient open.

It is nonempty.  For \(u\in\mathcal M_+\),
\[
m(\rho u)=\frac1{24}-\frac{\log3}{\rho^4},
\]
and \(J(\rho)\to0\), so all conditions in (50) hold at
\((\rho u,\psi_*)\) when \(\rho\) is large.

We next prove forward invariance by a first-exit argument.  Before a
hypothetical first exit, (29) prevents \(\rho\) from reaching
\(\rho_0\); (31) prevents the soft-margin quantity from falling to
its boundary value; and (48) prevents the value budget from reaching
\(\delta_v\).  If the direction \(u=\Phi/\rho\) were to reach
\(\partial U\), then (8), (30), and (32) would give
\[
m(\Phi)-J(\rho)
\le m(\Phi)
\le\Gamma(u)
\le\frac1{24}-4\Delta,
\]
contradicting the lower bound \(1/24-2\Delta\) in (50).  No first
exit is possible.

The solution is global.  Indeed, in \(\mathcal C\), (15)--(16) and
(6) give
\[
\|\dot\Phi\|
\le C\rho^3\mathcal A
\le3C\rho^3e^{-\mu\rho^4},
\tag{51}
\]
while the value variables stay in a bounded ball and have bounded
speed by (46).  Thus the raw parameters cannot escape to infinity
in finite time.  The vector field is smooth at every finite
parameter point, so the standard continuation criterion extends the
solution for every finite time.

Because \(\dot\rho>0\), either \(\rho\to\infty\) or it converges to a
finite number.  The latter is impossible: on the compact set
\(\overline U\), the normalized \(q_i\) are bounded above, so if
\(\rho\) were bounded then \(\mathcal A\) would be bounded below by
a positive constant.  Equation (29) would then bound \(\dot\rho\)
below by a positive constant.  Hence \(\rho\to\infty\).

Conditions (6) now imply
\[
H_{ij}(\Phi(t))\ge\mu\rho(t)^4\longrightarrow\infty
\quad\text{for all }i,j.
\]
Since \(S=-H\), this proves (41).  Equations (14) show that every
\(a_{ij}\) and \(b_{ij}\) tends to zero.  The value budget confines
\(g\) to the bounded negative interval in (44), so the delivered
coefficients also tend to zero.  Lemma \(\mathrm{lem:risk-identity}\)
then gives \(R\to1/2\), proving (42).

It remains to verify regularity and positive measure.  Let
\[
P_0=\frac{\mathbf1\mathbf1^\top}{3},\qquad P_\perp=I-P_0,
\]
and, for \(\varepsilon>0\), put
\[
E_\varepsilon=\frac{P_0}{\sqrt2}+\varepsilon P_\perp,\qquad
Q_\varepsilon=D_\varepsilon=\frac{P_0}{2}+\varepsilon P_\perp.
\tag{52}
\]
All three matrices in (52) are invertible.  After normalizing this
tuple and taking \(\varepsilon\) small, its direction lies in \(U\)
and is arbitrarily close to \(\mathcal M_+\).  Choose
\(\varepsilon\) small enough that its \(\Gamma\)-value exceeds
\(1/24-\Delta\), and then scale the tuple by a sufficiently large
\(\rho\).  Equations (32), (49), and (50) show that, with
\(K=-D_\varepsilon\) and \(\psi=\psi_*\), the resulting raw point
lies in \(\mathcal C\).  It also lies in \(\Theta_{\rm reg}\):
\(E,Q,K\) are invertible and \(c_g=3\).  Both
\(\mathcal C\) and \(\Theta_{\rm reg}\) are ambient open, so their
nonempty intersection is an ambient open subset of the full
parameter space.  It therefore has positive Lebesgue measure.
\(\square\)

# theorem thm:T

## statement

Define the finite regular-access set
\[
\Theta_{\rm reg}
=
\left\{
\theta:
\det E\,\det Q\,\det K\ne0,
\quad
c_g(\theta)>0
\right\},
\tag{5}
\]
where
\[
c_g(\theta)
=
\|OVu\|^2
+\|w\|^2\|Vu\|^2
+\|O^\top w\|^2\|u\|^2.
\tag{6}
\]

Let \(\lambda_\Theta\) denote Lebesgue measure on parameter space.

**Theorem T (maximal minimal-model target).** There is a
\(\lambda_\Theta\)-null set \(\mathcal N\subset\Theta_{\rm reg}\) such that, for
every \(\theta_0\in\Theta_{\rm reg}\setminus\mathcal N\), the solution of (2)
exists for all \(s\ge0\) and satisfies
\[
\max_{q\ne d}
\max\{|x_{qd}(\theta_s)-1|,|y_{qd}(\theta_s)|\}\to0.
\tag{T}
\]

Consequently, \(R(\theta_s)\to0\) and all six directed margins in (4) diverge.

Equivalently, (T) holds with probability one for every initialization law
\(\nu\ll\lambda_\Theta\) satisfying \(\nu(\Theta_{\rm reg})=1\).
The theorem is deliberately a prove-or-refute target.  A positive-measure regular
basin that violates (T) refutes it and must be retained as the missing-condition
witness.

## proof

The statement is false.  By Lemma \(\mathrm{lem:open-wrong-horn}\),
\[
\mathcal O=\mathcal C\cap\Theta_{\rm reg}
\]
is a nonempty ambient open set, hence
\(\lambda_\Theta(\mathcal O)>0\).  Every solution starting in
\(\mathcal O\) exists globally, but for every ordered pair \(q\ne d\),
\[
x_{qd}(\theta_s)\longrightarrow0,\qquad
y_{qd}(\theta_s)\longrightarrow0.
\]
Therefore
\[
\max_{q\ne d}
\max\{|x_{qd}(\theta_s)-1|,|y_{qd}(\theta_s)|\}
\longrightarrow1,
\]
and \(R(\theta_s)\to1/2\), not zero.  Any exceptional set containing
all these failing initial points would have to contain the
positive-measure open set \(\mathcal O\), so it could not be null.

The probability formulation is likewise false: take any bounded open
ball whose closure lies in \(\mathcal O\), and normalize Lebesgue
measure on that ball.  This gives an initialization law absolutely
continuous with respect to \(\lambda_\Theta\), supported on
\(\Theta_{\rm reg}\), for which (T) fails with probability one.

The precise missing-condition witness is the horn (50): any corrected
convergence theorem must at least exclude this large-norm,
negative-gain, all-negative-score max-margin basin.  The simpler
condition \(g(\theta_0)\ge0\) excludes this particular witness, but no
sufficiency claim for that stronger condition is proved here.
\(\square\)
