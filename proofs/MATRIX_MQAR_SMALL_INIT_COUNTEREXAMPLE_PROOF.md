# Small-initialization matrix MQAR: a positive-density access-collapse basin

The proposed density theorem is false. The obstruction is already present at
the score origin. After an open cone of value-factor directions selects
positive gain, score factors with \(Q^{\top}K\) negatively oriented can lower
the loss by shrinking the contrast part of \(E\) to zero. They converge to a
finite, normally attracting manifold on which every delivered target and
distractor coefficient equals \(1/2\). This gives risk \(1/4\), and the basin
contains a fixed open set of rescaled directions. It therefore has positive
lower density at the initialization origin.

The proof is finite-dimensional and self-contained. No external theorem is
used.

# definition def:frozen-mqar

## statement

The frozen population has \(C=d=3\), two memory slots, and
\(N=C(C-1)=6\) ordered target--distractor pairs.  For each \(q\ne d\), both
memory-slot orderings occur and the target and distractor values independently
range over \(\{-1,+1\}\).  Thus there are \(6\cdot2\cdot4=48\) equally
weighted episodes, and the label is the value attached to the target key \(q\).

The active Euclidean parameter is
\[
\theta=(E,Q,K,O,z,w)\in
(\mathbb R^{3\times3})^4\times(\mathbb R^3)^2\cong\mathbb R^{42}.
\]
Put
\[
B=Q^\top K,\qquad S=EBE^\top,\qquad g=w^\top Oz.
\tag{2}
\]
For \(q\ne d\), define the exact-softmax coefficients
\[
D_{qd}=1+e^{S_{qq}}+e^{S_{qd}},\qquad
a_{qd}=\frac{e^{S_{qq}}}{D_{qd}},\qquad
b_{qd}=\frac{e^{S_{qd}}}{D_{qd}}.
\tag{3}
\]
The fixed \(1\) in \(D_{qd}\) is the zero-valued query-self token.  On an
episode with values \(v_q,v_d\), the prediction and delivered coefficients are
\[
f_\theta=g(a_{qd}v_q+b_{qd}v_d),\qquad
x_{qd}=ga_{qd},\qquad y_{qd}=gb_{qd}.
\]
The delivered and target kernels are
\[
\kappa(\theta)=((x_{qd},y_{qd}))_{q\ne d},\qquad
\kappa^*=((1,0))_{q\ne d}.
\tag{4}
\]
The population half-squared risk is
\[
R(\theta)=\frac1{2N}\sum_{q\ne d}
\bigl[(x_{qd}-1)^2+y_{qd}^2\bigr]
=\frac1{2N}\|\kappa(\theta)-\kappa^*\|_2^2.
\tag{5}
\]
Let \(\phi_t(\theta_0)\) be the maximal negative-gradient-flow solution and
define, for use throughout the intervening lemmas,
\[
\mathcal F=\{\theta_0:\phi_t(\theta_0)\text{ is not global, or }
\limsup_{t\to\infty}R(\phi_t(\theta_0))>0\}.
\tag{20-def}
\]
The name \(T_{\rm small}\) denotes the claim that the Lebesgue density of
\(\mathcal F\) at the origin is zero. The final theorem restates the original
target in full.

Write
\[
G_S=\nabla_SR,\qquad \gamma=\partial_gR,\qquad
G_B=E^\top G_SE.
\]
The Euclidean factor gradients are
\[
G_E=G_SEB^\top+G_S^\top EB,\qquad
G_Q=KG_B^\top,\qquad G_K=QG_B,
\tag{9}
\]
\[
G_w=\gamma Oz,\qquad G_O=\gamma wz^\top,\qquad
G_z=\gamma O^\top w.
\tag{10}
\]
Consequently gradient flow preserves the four balance tensors
\[
I_{QK}=QQ^\top-KK^\top,\qquad
I_{EQK}=E^\top E-Q^\top Q-K^\top K,
\tag{11}
\]
\[
J_L=ww^\top-OO^\top,\qquad
J_R=O^\top O-zz^\top.
\tag{12}
\]

Define
\[
\mathsf A(S)=\frac1N\sum_{q\ne d}(a_{qd}^2+b_{qd}^2),
\qquad
\mathsf B(S)=\frac1N\sum_{q\ne d}a_{qd}.
\]
Then
\[
\gamma=\mathsf A(S)g-\mathsf B(S),\qquad
\dot g=c_g\bigl(\mathsf B(S)-\mathsf A(S)g\bigr),
\tag{13}
\]
where
\[
c_g=\|Oz\|^2+\|w\|^2\|z\|^2+\|O^\top w\|^2
=\|\nabla_{(w,O,z)}g\|^2.
\tag{14}
\]

For \(e=\kappa-\kappa^*\), \(J_\kappa=D\kappa(\theta)\), and \(e\ne0\),
the task-kernel access is
\[
\mathsf A_\kappa(\theta,e)
=\frac{\|J_\kappa(\theta)^\top e\|_2^2}{\|e\|_2^2}.
\tag{15}
\]
Along gradient flow,
\[
\dot R=-\frac1{N^2}\|J_\kappa^\top e\|_2^2
=-\frac2N\mathsf A_\kappa R,
\tag{16}
\]
and hence, as long as \(R(0)>0\),
\[
R(t)=R(0)\exp\!\left[
-\frac2N\int_0^t\mathsf A_\kappa(\theta_s,e_s)\,ds\right].
\tag{17}
\]

For a scaled initialization \(\theta=\varepsilon\bar\theta\), homogeneity
gives exactly
\[
S=\varepsilon^4\bar E\bar Q^\top\bar K\bar E^\top,\qquad
g=\varepsilon^3\bar w^\top\bar O\bar z,\qquad
c_g=\varepsilon^4\bar c_g,
\tag{21}
\]
where
\(\bar c_g=\|\bar O\bar z\|^2+\|\bar w\|^2\|\bar z\|^2+
\|\bar O^\top\bar w\|^2\).
At initialization,
\[
\dot g=O(\varepsilon^4),\qquad \dot S=O(\varepsilon^9).
\tag{22}
\]
On fast time \(\tau=\varepsilon t\), the leading value system is
\[
\bar w'=\frac13\bar O\bar z,\qquad
\bar O'=\frac13\bar w\bar z^\top,\qquad
\bar z'=\frac13\bar O^\top\bar w,
\tag{23}
\]
with
\[
\frac d{d\tau}(\bar w^\top\bar O\bar z)=\frac13\bar c_g\ge0.
\tag{24}
\]
Finally,
\[
G_S(0,g)=\frac g{54}(\mathbf1\mathbf1^\top-5I)
+\frac{g^2}{162}(\mathbf1\mathbf1^\top+I),
\tag{29}
\]
so, for \(P_\perp=I-\mathbf1\mathbf1^\top/3\),
\[
G_S(0,3/2)=-\frac18P_\perp.
\tag{30}
\]

# lemma lem:population-algebra

## statement

For the frozen \(C=d=3,m=2\) population of Definition
\(\mathrm{def:frozen-mqar}\), the fully displayed identities (5), (9)--(17),
(21)--(24), and (29)--(30) are exact. In particular,
with \(N=6\), \(e=\kappa-\kappa^*\), and \(J_\kappa=D\kappa\),
\[
 R=\frac1{2N}\|e\|^2,
 \qquad
 \dot R=-\frac1{N^2}\|J_\kappa^{\top}e\|^2,
\tag{34}
\]
and
\[
G_S(0,g)=\frac g{54}({\bf1}{\bf1}^{\top}-5I)
 +\frac{g^2}{162}({\bf1}{\bf1}^{\top}+I),
\qquad
G_S(0,3/2)=-\frac18P_\perp.
\tag{35}
\]

An independent complete-population and forward-mode differentiation check has
maximum discrepancy \(1.7763568394002505\cdot10^{-15}\).

## proof

Fix an ordered target--distractor pair \((q,d)\). If
\(v_q,v_d\) are independent Rademacher variables, then
\[
f-Y=(ga_{qd}-1)v_q+gb_{qd}v_d.
\]
The mixed term has expectation zero. Averaging the resulting two squares over
the six ordered pairs proves
\[
R=\frac1{12}\sum_{q\ne d}
   \{(ga_{qd}-1)^2+(gb_{qd})^2\},
\]
which is (5).

Put \(B=Q^{\top}K\). The differentials
\[
dS=dE\,BE^{\top}+E(dQ^{\top}K+Q^{\top}dK)E^{\top}
     +EB\,dE^{\top},
\qquad
dg=dw^{\top}Oz+w^{\top}dO\,z+w^{\top}O\,dz
\]
give, by the Frobenius chain rule,
\[
G_E=G_SEB^{\top}+G_S^{\top}EB,\quad
G_Q=K(E^{\top}G_SE)^{\top},\quad
G_K=Q(E^{\top}G_SE),
\]
and
\[
G_w=\gamma Oz,\qquad G_O=\gamma wz^{\top},\qquad
G_z=\gamma O^{\top}w.
\]
Set \(M=G_B=E^\top G_SE\).  Under negative gradient flow,
\[
\dot Q=-KM^\top,\qquad \dot K=-QM.
\]
Therefore
\[
\begin{aligned}
\frac d{dt}(QQ^\top)
 &=-KM^\top Q^\top-QMK^\top,\\
\frac d{dt}(KK^\top)
 &=-QMK^\top-KM^\top Q^\top,
\end{aligned}
\]
which proves \(\dot I_{QK}=0\).  Likewise
\[
\begin{aligned}
\frac d{dt}(E^\top E)
={}&-(BM^\top+B^\top M+MB^\top+M^\top B),\\
\frac d{dt}(Q^\top Q)
={}&-(MB^\top+BM^\top),\\
\frac d{dt}(K^\top K)
={}&-(M^\top B+B^\top M).
\end{aligned}
\]
Their indicated difference is zero, proving \(\dot I_{EQK}=0\).
For the value factors,
\[
\begin{aligned}
\frac d{dt}(ww^\top)
 &=-\gamma(Ozw^\top+wz^\top O^\top)
  =\frac d{dt}(OO^\top),\\
\frac d{dt}(O^\top O)
 &=-\gamma(zw^\top O+O^\top wz^\top)
  =\frac d{dt}(zz^\top).
\end{aligned}
\]
Thus \(\dot J_L=\dot J_R=0\) as claimed; no scalar balance or exact
initial balancing assumption has been used.

Differentiating the risk with respect to \(g\) gives
\[
\gamma=\frac1N\sum_{q\ne d}
 \{g(a_{qd}^2+b_{qd}^2)-a_{qd}\}
       =\mathsf A(S)g-\mathsf B(S).
\]
Moreover
\[
\|\nabla_{(w,O,z)}g\|^2
=\|Oz\|^2+\|w\|^2\|z\|^2+\|O^{\top}w\|^2=c_g,
\]
so \(\dot g=-\gamma c_g=c_g(\mathsf B-\mathsf A g)\).
Since \(\nabla_\theta R=N^{-1}J_\kappa^{\top}e\), (34) follows.
More explicitly,
\[
\dot R=\langle\nabla_\theta R,\dot\theta\rangle
=-\|\nabla_\theta R\|^2
=-\frac1{N^2}\|J_\kappa^\top e\|^2.
\]
Dividing by \(R=\|e\|^2/(2N)\) gives
\(\dot R/R=-(2/N)\mathsf A_\kappa\); integration gives (17).  This proves
(15)--(17) directly from the defined kernel map.

Under \(\theta=\varepsilon\bar\theta\), the two factor maps have the exact
degrees
\[
S=\varepsilon^4\bar E\bar Q^{\top}\bar K\bar E^{\top},
\qquad g=\varepsilon^3\bar w^{\top}\bar O\bar z,
\qquad c_g=\varepsilon^4\bar c_g.
\]
At \(S=O(\varepsilon^4),g=O(\varepsilon^3)\), one has
\(\gamma=-1/3+o(1)\). The raw score gradient is \(O(\varepsilon^6)\);
differentiating the degree-four map \(S\) then gives
\(\dot S=O(\varepsilon^9)\). The value gradient is
\(O(\varepsilon^2)\), so on \(\tau=\varepsilon t\)
\[
\bar w'=\frac13\bar O\bar z,\qquad
\bar O'=\frac13\bar w\bar z^{\top},\qquad
\bar z'=\frac13\bar O^{\top}\bar w.
\]
Taking the derivative of \(\bar w^{\top}\bar O\bar z\) proves (24).

Finally, for the pair loss
\(\ell_{qd}=\tfrac12[(ga_{qd}-1)^2+(gb_{qd})^2]\), put
\(u=S_{qq}\), \(v=S_{qd}\). The two softmax derivatives are
\[
\begin{array}{ll}
\partial_u a=a(1-a),&\partial_u b=-ab,\\
\partial_v a=-ab,&\partial_v b=b(1-b).
\end{array}
\]
At \(S=0\), \(a=b=1/3\), so the contribution after the \(1/N=1/6\)
population weight is
\[
\frac1N\partial_u\ell_{qd}=\frac{g^2}{162}-\frac g{27},\qquad
\frac1N\partial_v\ell_{qd}=\frac g{54}+\frac{g^2}{162}.
\]
Each diagonal entry \(S_{qq}\) occurs in two ordered pairs and each
off-diagonal entry \(S_{qd}\) in one. Hence the diagonal and off-diagonal
entries of \(G_S(0,g)\) are respectively
\[
-\frac{2g}{27}+\frac{g^2}{81},
\qquad
\frac g{54}+\frac{g^2}{162},
\]
which is exactly the first identity in (35).
At \(g=3/2\), its \({\bf1}{\bf1}^{\top}\) coefficient cancels exactly and
its \(P_\perp\) coefficient is \(-1/8\). Thus no \(P_0\) component is
present.

The independent audit is scripts/n0_small_initialization_audit.py. It
enumerates all \(48\) episodes, differentiates that computation with float64
forward-mode dual numbers, and checks the raw gradients, all balances, (34),
the gain identity, homogeneity, and (35). Its largest reported discrepancy is
the number stated above. This also reconciles the formulas with the
predecessor: its risk and raw chain rule remain valid, while the corrected
score-origin calculation is (35). \(\square\)

# lemma lem:quantifiers

## statement

The failure set \(\mathcal F\) in (20-def) is Borel. The density statement
\(T_{\rm small}\) is equivalent to
\[
\Pr_{\xi\sim\nu}\{\varepsilon\xi\in\mathcal F\}\longrightarrow0
\quad(\varepsilon\downarrow0)
\tag{36}
\]
for every fixed probability law \(\nu\ll\lambda_\Theta\). The raywise
statement \(T_{\rm ray}\) implies (36), but the converse is false in dimension
\(42\).

## proof

Let \(D_t\) be the set of initial points whose maximal solution exists through
time \(t\). Smooth ODE dependence makes \(D_t\) open and
\(\theta\mapsto R(\phi_t(\theta))\) continuous on it. The global set is the
Borel set \(\mathcal G=\bigcap_{n\ge1}D_n\). Since risk is nonincreasing on a
global trajectory, its limit exists, and
\[
\mathcal F=(\Theta\setminus\mathcal G)\cup
 \left[\mathcal G\cap
  \bigcup_{m\ge1}\bigcap_{n\ge1}
  \{\theta\in D_n:R(\phi_n(\theta))\ge1/m\}\right].
\tag{37}
\]
This proves measurability.

Suppose first that the density in \(T_{\rm small}\) is zero. Write
\(E_\varepsilon=\{\xi:\varepsilon\xi\in\mathcal F\}\). For every fixed
\(M\),
\[
\lambda(E_\varepsilon\cap B_M)
=\varepsilon^{-42}\lambda(\mathcal F\cap B_{\varepsilon M})=o(1).
\]
Absolute continuity of the Lebesgue integral therefore gives
\(\nu(E_\varepsilon\cap B_M)\to0\). First choose \(M\) with
\(\nu(B_M^c)\) small and then let \(\varepsilon\downarrow0\); this proves
(36). Conversely, apply (36) to normalized Lebesgue measure on \(B_1\).
Its failure probability is exactly
\(\lambda(\mathcal F\cap B_\varepsilon)/\lambda(B_\varepsilon)\).

Raywise eventual success gives pointwise convergence
\({\bf1}_{E_\varepsilon}(\xi)\to0\) for almost every \(\xi\), so dominated
convergence proves (36). To see strictness, take spherical caps
\(A_n\subset S^{41}\) whose surface fractions tend to zero but whose limsup is
all of \(S^{41}\): for each \(k\), list once all caps in a finite
radius-\(1/k\) cover. On the annulus
\(2^{-(n+1)}<\|x\|\le2^{-n}\), retain exactly the directions in \(A_n\).
The resulting set has density zero because each shrinking-ball density is a
weighted average of the cap fractions in its inner annuli. Every ray
nevertheless meets it in infinitely many annuli. Thus density zero does not
imply eventual raywise avoidance. \(\square\)

# lemma lem:value-clock

## statement

Let \(\psi=(w,O,z)\in\mathbb R^{15}\) and
\(h(\psi)=w^{\top}Oz\). While \(\gamma<0\), the value-factor trajectory is a
positive time change of
\[
\frac{d\psi}{ds}=\nabla h(\psi).
\tag{38}
\]
There is a nonempty open angular set \(U_v\subset S^{14}\), containing the
balanced positive rank-one manifold
\[
\mathcal V_+=\left\{
 (u/\sqrt3,uv^{\top}/\sqrt3,v/\sqrt3):u,v\in S^2
\right\},
\tag{39}
\]
with the following uniform property. If
\(\psi_0=\varepsilon\rho_0n_0\), where
\(n_0\in U_v\) and \(\rho_0\) ranges in a fixed compact subinterval of
\((0,\infty)\), then the orbit (38) reaches every prescribed gain
\(g_c>0\). Before that hitting time it remains in a fixed compact angular cone
and
\[
\int_0^{s_c} h(\psi_s)\,ds\le C(g_c,U_v).
\tag{40}
\]

For comparison, on a frozen score path \(S=0\), the balanced negative ray
\[
w=-re_1,\qquad O=re_1e_1^\top,\qquad z=re_1
\tag{25}
\]
is invariant and satisfies
\[
\dot r=-\left(\frac13+\frac{2r^3}{9}\right)r^2<0.
\tag{26}
\]
It has \(g=-r^3\uparrow0\) and does not enter \(U_v\). No global
classification of directions outside \(U_v\) is needed for the
positive-density counterexample.

## proof

The value equations are
\[
\dot w=-\gamma Oz,\qquad \dot O=-\gamma wz^{\top},\qquad
\dot z=-\gamma O^{\top}w.
\]
Thus \(ds/dt=-\gamma\) gives (38). On \(S^{14}\), Cauchy--Schwarz and
arithmetic--geometric mean give
\[
h(w,O,z)\le\|w\|\|O\|_F\|z\|
 \le\frac1{3\sqrt3}.
\tag{41}
\]
Equality holds exactly on (39). Write \(\psi=\rho n\) and use angular time
\(d\sigma=\rho\,ds\). Euler's identity gives
\[
\frac{d\rho}{ds}=3\rho^2h(n),\qquad
\frac{dn}{d\sigma}=\nabla_{S^{14}}h(n),\qquad
\frac{dh(n)}{d\sigma}=\|\nabla_{S^{14}}h(n)\|^2.
\tag{42}
\]
Choose \(U_v\) to be the union of the components surrounding
\(\mathcal V_+\) of a sufficiently high strict superlevel set of \(h\).
The equality classification in (41) and compactness make its closure compact,
contained in \(\{h\ge h_0>0\}\), and forward invariant by (42). Hence
\(\rho\) increases until \(\rho^3h(n)=g_c\). Since
\(h_0\le h(n)\le1/(3\sqrt3)\), both the hitting radius and
\[
\int_0^{s_c}h(\psi_s)\,ds
 \le C\int_{\rho(0)}^{\rho(s_c)}
       \rho^3\frac{d\rho}{\rho^2}
\]
are uniformly bounded. This proves (40).

For (25), \(\gamma=2g/9-1/3=-1/3-2r^3/9\). Substitution in the three
value-factor equations gives the same scalar equation (26) for all three
blocks. This verifies the comparison trajectory without making an unused
claim about every other direction. \(\square\)

# lemma lem:fast-transfer

## statement

Fix a compact subset \(V\Subset U_v\), a compact interval of initial value
radii, and \(g_c<3/2\) sufficiently close to \(3/2\). Fix also a compact set
of normalized score triples
\(\bar\Phi=(\bar E,\bar Q,\bar K)\). Uniformly on these sets, the full flow
from
\[
(E,Q,K)_0=\varepsilon(\bar E,\bar Q,\bar K),\qquad
\psi_0=\varepsilon\bar\psi_0
\tag{45}
\]
has, for all sufficiently small \(\varepsilon\), a first time \(T_\varepsilon\)
at which
\[
g(T_\varepsilon)=g_c,\qquad
\|(E,Q,K)(T_\varepsilon)-\varepsilon\bar\Phi\|=O(\varepsilon^3),
\qquad \|S(T_\varepsilon)\|=O(\varepsilon^4).
\tag{46}
\]
If the three normalized score matrices are invertible, score access does not
collapse before \(T_\varepsilon\). The same proof gives the N3 hitting section
\(g=1\).

## proof

For \(S\) in a fixed small neighborhood of zero, write
\[
\gamma=\mathsf A(S)\{g-g_{\rm opt}(S)\},\qquad
g_{\rm opt}(S)=\mathsf B(S)/\mathsf A(S).
\]
Continuity and \(g_{\rm opt}(0)=3/2\) allow \(g_c<3/2\) and a score
neighborhood on which \(g_{\rm opt}\ge g_c+c_0\) and
\(\mathsf A\ge a_0>0\). Thus \(-\gamma\ge a_0c_0\) until \(g=g_c\).
The value orbit is consequently the time change in Lemma
\(\mathrm{lem:value-clock}\) and reaches that section in finite physical time.

For \(0\le g\le g_c\) and bounded \(S\), direct inspection of the softmax
derivative gives \(\|G_S\|\le Cg\). Every raw score gradient is cubic in the
score factors, so a bootstrap under \(\|\Phi\|\le C_1\varepsilon\) gives
\[
\left\|\frac{d\Phi}{ds}\right\|
 \le C\varepsilon^3 g.
\]
Equation (40) now yields total score motion \(O(\varepsilon^3)\). This closes
the bootstrap and proves (46). Singular values of invertible
\(\varepsilon\bar E,\varepsilon\bar Q,\varepsilon\bar K\) are changed by only
\(O(\varepsilon^3)\), so none reaches zero. Taking \(g_c=1\) proves (32), and
(27) follows from \(S(T_\varepsilon)\to0\). \(\square\)

# lemma lem:entrance-region

## statement

The set
\[
\mathcal U=\{\theta:g\ge0,\ R\le7/18\}
\tag{47}
\]
is forward invariant. Every trajectory in Lemma
\(\mathrm{lem:fast-transfer}\) enters it at its \(g=1\) hitting time for all
sufficiently small \(\varepsilon\).

## proof

At \(g=0\), (13) gives
\(\dot g=c_g\mathsf B(S)\ge0\); uniqueness therefore prevents crossing from
\(g\ge0\) to \(g<0\). Risk dissipation makes every risk sublevel forward
invariant. At the hitting section, \(S=o(1)\), and (27) gives
\(R\to5/18<7/18\). \(\square\)

# lemma lem:uniform-wrong-manifold

## statement

Let
\[
p=\frac{{\bf1}}{\sqrt3},\qquad P_0=pp^{\top},\qquad P=I-P_0.
\]
There is a \(35\)-dimensional manifold of exact critical points
\[
\mathcal M=\left\{
\begin{array}{l}
E=pa^{\top},\quad B=Q^{\top}K,\quad
\beta=a^{\top}Ba,\\[1mm]
g=g_*(\beta):=1+\frac12e^{-\beta/3},\quad c_g>0
\end{array}\right\}.
\tag{48}
\]
Every point of \(\mathcal M\) has
\[
S=\beta P_0,\qquad x_{qd}=y_{qd}=\frac12,\qquad
R=\frac14,\qquad G_S=-\frac18P.
\tag{49}
\]
At a point of (48) with \(E=0\), the Hessian quadratic form is
\[
D^2R[\delta\theta,\delta\theta]
=\frac29(\delta g)^2
-\frac14\operatorname{tr}
 \left((\delta E)^{\top}P(\delta E)\,
       \operatorname{sym}(Q^{\top}K)\right).
\tag{50}
\]
Consequently, if \(\operatorname{sym}(Q^{\top}K)\prec0\), the loss Hessian
has seven positive normal directions and a \(35\)-dimensional kernel equal to
\(T\mathcal M\). This positivity persists on a sufficiently small patch of
(48).

## proof

If \(E=pa^{\top}\), then
\(S=p(a^{\top}Ba)p^{\top}=\beta P_0\), so every memory score equals
\(\beta/3\). Put
\[
t(\beta)=\frac{e^{\beta/3}}{1+2e^{\beta/3}}.
\]
All six pairs have \(a_{qd}=b_{qd}=t\), and
\[
R=(gt)^2-gt+\frac12=(gt-\tfrac12)^2+\frac14.
\tag{51}
\]
Its value optimum is \(g_*=1/(2t)=1+\tfrac12e^{-\beta/3}\).
At that optimum \(\gamma=0\). For one ordered pair the diagonal and
off-diagonal score derivatives are respectively \(-1/24\) and \(1/24\);
each diagonal occurs twice. Hence \(G_S=-P/8\), independently of \(\beta\).
Since \(PE=0\),
\[
G_SE=G_S^{\top}E=0,\qquad E^{\top}G_SE=0.
\]
All score gradients vanish, and \(\gamma=0\) makes all value gradients
vanish. This proves (48)--(49). The free dimensions are \(3\) for \(a\),
\(18\) for \(Q,K\), and \(14\) for the regular level set of \(g\), totaling
\(35\).

At \(E=0\), a variation \(F=\delta E\) produces
\[
S(\theta+t\delta\theta)=t^2FQ^{\top}KF^{\top}+O(t^3).
\]
At \(S=0,g=3/2\), one has \(R_{gg}=2/9\) and \(G_S=-P/8\).
The second-order chain rule therefore gives
\[
D^2R=\frac29(\delta g)^2
+2\langle-P/8,FQ^{\top}KF^{\top}\rangle,
\]
which is (50), because the skew part has zero trace against
\(F^{\top}PF\). If the symmetric part of \(Q^{\top}K\) is negative definite,
the second term is positive exactly when \(PF\ne0\). Its kernel consists of
\(PF=0\), \(\delta g=0\), and arbitrary \(\delta Q,\delta K\), precisely the
\(35\)-dimensional tangent space of (48). Continuity, together with the fact
that every tangent vector remains in the Hessian kernel because the gradient
vanishes identically on \(\mathcal M\), proves persistence of rank seven and
normal positivity on a small patch. \(\square\)

# lemma lem:leading-collapse

## statement

At \(g=3/2,S=0\), put
\((E,Q,K)=\varepsilon(\bar E,\bar Q,\bar K)\) and use slow time
\(u=\varepsilon^2t\). The limiting score flow is
\[
\bar E'=\frac18P\bar E(\bar B^{\top}+\bar B),\qquad
\bar Q'=\frac18\bar K C,\qquad
\bar K'=\frac18\bar Q C,
\tag{52}
\]
where \(\bar B=\bar Q^{\top}\bar K\) and
\(C=\bar E^{\top}P\bar E\).
It has a nonempty ambient-open collapse basin. More explicitly, for
\(\bar D=-\bar K\), all initial points satisfying
\[
\|\bar Q-I\|+\|\bar D-I\|<\delta,\qquad
\|P\bar E\|<\eta
\tag{53}
\]
belong to that basin when \(\delta,\eta>0\) are sufficiently small, and
\[
P\bar E(u)\longrightarrow0,\qquad
\bar Q(u),\bar D(u)\ \hbox{converge near }I.
\tag{54}
\]
Thus the wrong angular set is attracting and has positive angular measure.

## proof

Equation (52) follows by inserting \(G_S=-P/8\) in (9), scaling the three
score factors by \(\varepsilon\), and dividing time by \(\varepsilon^{-2}\).
Let \(U=P\bar E\). In the variables \((U,\bar Q,\bar D)\), it becomes
\[
U'=-\frac18U(\bar D^{\top}\bar Q+\bar Q^{\top}\bar D),\quad
\bar Q'=-\frac18\bar D\,U^{\top}U,\quad
\bar D'=-\frac18\bar Q\,U^{\top}U.
\tag{55}
\]
As long as \(\bar Q,\bar D\) remain in a sufficiently small fixed ball about
\(I\), the matrix in parentheses is at least the identity. Hence
\[
\frac d{du}\|U\|_F^2\le-\frac14\|U\|_F^2,\qquad
\|\bar Q'\|+\|\bar D'\|\le C\|U\|_F^2.
\tag{56}
\]
The total drift of \((\bar Q,\bar D)\) is therefore at most \(C\eta^2\).
Choose \(\eta\) so that this is smaller than half the buffer in (53).
A first-exit argument closes the bootstrap, proves exponential decay of \(U\),
and makes both remaining derivatives integrable. This proves (54). Since all
inequalities in (53) can be strict, the basin is ambient open. \(\square\)

# lemma lem:uniform-tube

## statement

There are constants
\[
\beta_*>0,\quad \rho_*>0,\quad M_*>0,\quad
\delta_g>0,\quad \eta>0,\quad \varepsilon_0>0
\tag{57}
\]
and nested value-factor sets
\(K_v^{\rm in}\Subset V_v^{\rm out}\), with \(K_v^{\rm in}\) compact,
\(V_v^{\rm out}\) open and bounded, and \(c_g\) bounded below on
\(\overline{V_v^{\rm out}}\), such that the following holds for every
\(0<\varepsilon<\varepsilon_0\).
Suppose
\[
\|Q/\varepsilon-I\|+\|K/\varepsilon+I\|<\rho_*,\qquad
\operatorname{sym}\!\left((Q/\varepsilon)^{\top}
                           (K/\varepsilon)\right)\preceq-3\beta_* I,
\qquad
\|E/\varepsilon\|<M_*,\qquad
\|PE/\varepsilon\|<\eta,
\tag{58}
\]
\(\psi\in K_v^{\rm in}\), and, with \(a=E^{\top}p\),
\[
\left|g-\left(1+\frac12e^{-a^{\top}Q^{\top}Ka/3}\right)\right|<\delta_g.
\]
Then the exact full gradient-flow solution
is global and converges to a point of \(\mathcal M\) in (48). In particular,
\[
R(t)\to\frac14,\qquad x_{qd}(t),y_{qd}(t)\to\frac12
\quad(q\ne d).
\tag{59}
\]
The constants in (57) are independent of \(\varepsilon\); thus (58) is an
open tube of width proportional to the score scale, not a shrinking
lower-dimensional stable manifold.

## proof

We give the parameter-uniform normal estimate. It is also a direct proof of
the local Morse--Bott assertion, so no stable-manifold result is being used as
a black box.

Choose first a compact regular value-factor neighborhood near the positive
rank-one \(g=3/2\) level, and then a slightly larger bounded open neighborhood:
these are \(K_v^{\rm in}\Subset V_v^{\rm out}\). Euler's identity
\(\langle\nabla g(\psi),\psi\rangle=3g(\psi)\) and compactness allow them to
be chosen so that \(c_g=\|\nabla g\|^2\ge c_v>0\) throughout
\(\overline{V_v^{\rm out}}\). Choose fixed inner score radii
\(\rho_*,M_*\); all estimates below are made on the corresponding outer
patch with radii \(2\rho_*,2M_*\), gain radius \(2\delta_g\), contrast radius
\(2\eta\), and value set \(V_v^{\rm out}\). Thus the hypotheses (58) have
fixed positive buffers before any first exit.

Write
\[
E=pa^{\top}+F,\qquad p^{\top}F=0,\qquad B=Q^{\top}K,
\qquad s=a^{\top}Ba,
\]
and set \(r=g-g_*(s)\). Relative to the point of \(\mathcal M\) with the
same \(a,Q,K\), the score perturbation is
\[
\Delta S=FBa\,p^{\top}+pa^{\top}BF^{\top}+FBF^{\top}.
\tag{60}
\]
The first two terms are orthogonal to \(G_S=-P/8\). Under (58),
\(\operatorname{sym}B\preceq-c_0\varepsilon^2I\),
\(\|a\|=O(\varepsilon)\), and
\[
\langle-P/8,FBF^{\top}\rangle
\ge \frac{c_0}{8}\varepsilon^2\|F\|_F^2.
\tag{61}
\]
Taylor's formula for the smooth quotient risk, uniformly on the compact
patch, now gives
\[
c_1\{r^2+\varepsilon^2\|F\|_F^2\}
\le R-\frac14
\le C_1\{r^2+\varepsilon^2\|F\|_F^2\}.
\tag{62}
\]
Indeed the square of the part of (60) linear in \(F\) is
\(O(\varepsilon^6\|F\|^2)\), which is absorbed by (61); all other remainders
are absorbed after reducing \(\delta_g,\eta,\varepsilon_0\).

We now prove that the tube has scale-independent normalized width. Set
\[
X=E/\varepsilon,\quad Y=Q/\varepsilon,\quad Z=K/\varepsilon,\quad
C=Y^{\top}Z,\quad f=PX,
\]
and use slow time \(\sigma=\varepsilon^2t\). With \(G=G_S\), the score
equations become exactly
\[
X'=-GXC^{\top}-G^{\top}XC,\quad
Y'=-ZX^{\top}G^{\top}X,\quad
Z'=-YX^{\top}GX,\quad
\psi'=-\frac{\gamma}{\varepsilon^2}\nabla g(\psi).
\tag{63}
\]
Let
\[
\bar a=X^{\top}p=a/\varepsilon,\qquad
X_0=P_0X=p\bar a^{\top},
\qquad S_0=\varepsilon^4X_0CX_0^{\top}=sP_0,
\]
which is uniform. Permutation symmetry gives
\[
G_S(S_0,g)=\alpha P+\lambda P_0.
\]
Put \(\gamma_0=\partial_gR(S_0,g)\). On \(\gamma_0=0\),
\(\alpha=-1/8,\lambda=0\), by (49). Smoothness on the compact patch,
together with
\[
\|S-S_0\|\le C\varepsilon^4\|f\|,\qquad
|\gamma_0-\gamma|\le C\varepsilon^4\|f\|,
\qquad \gamma=2t_0^2r+O(\varepsilon^4\|f\|),
\]
where \(t_0\) is the common attention weight at \(S_0\),
therefore gives
\[
|\alpha+1/8|+|\lambda|
\le C(|\gamma|+\varepsilon^4\|f\|),\qquad
\|G_S(S,g)-G_S(S_0,g)\|
\le C\varepsilon^4\|f\|.
\tag{64}
\]

Put \(H=\operatorname{sym}C\). We bootstrap the weaker buffered inequality
\(H\preceq-\beta_* I\). Projecting the first equation in (63), using (64), and
differentiating
\(\gamma=\mathsf A(S)g-\mathsf B(S)\) give the upper Dini inequalities
\[
\frac d{d\sigma}\|f\|^2\le-\kappa\|f\|^2,\qquad
D^+|\gamma|
\le-\frac{c}{\varepsilon^2}|\gamma|
+C\varepsilon^4\|f\|.
\tag{65}
\]
Here \(\kappa,c>0\) are uniform. The first leading term is explicitly
\[
2\left\langle f,\frac14fH\right\rangle
=\frac12\operatorname{tr}(f^{\top}fH);
\]
all its errors are bounded by the right side of (64). In the second
inequality the value contribution is exactly
\(-\mathsf A(S)c_g\gamma/\varepsilon^2\), while score motion contributes
\(O(\varepsilon^4\|f\|)+O(\varepsilon^4|\gamma|)\); the latter term is
absorbed into the leading negative coefficient after reducing
\(\varepsilon_0\).

The remaining three equations in (63) and (64) give
\[
\|Y'\|+\|Z'\|+\|(p^{\top}X)'\|
\le C\{\|f\|^2+|\gamma|+\varepsilon^4\|f\|\}.
\]
Indeed, for \(G_0=\alpha P+\lambda P_0\),
\[
X^{\top}G_0X=\alpha f^{\top}f+\lambda\bar a\bar a^{\top},
\]
and the nonsymmetric remainder, including the skew part of
\(E^{\top}G_SE\), is \(O(\varepsilon^4\|f\|)\). Thus the displayed estimate
controls arbitrary nearby \(Q,K\), not only the slice \(K=-Q\).
Integrating (65) and this bound yields
\[
\begin{aligned}
&\int_0^\infty\|f\|^2d\sigma\le C\|f(0)\|^2,\qquad
\int_0^\infty\|f\|d\sigma\le C\|f(0)\|,\\
&\int_0^\infty|\gamma|d\sigma
\le C\varepsilon^2|\gamma(0)|
+C\varepsilon^6\|f(0)\|,\\
&\operatorname{Var}(Y,Z,p^{\top}X)
\le C\{\|f(0)\|^2+\varepsilon^2|\gamma(0)|
              +\varepsilon^4\|f(0)\|\},\\
&\operatorname{Var}(\psi)
\le C\{|\gamma(0)|+\varepsilon^4\|f(0)\|\}.
\end{aligned}
\tag{66}
\]

Choose the fixed outer patch and \(\beta_*\) first, then the normalized
contrast radius \(\eta\), then the gain radius \(\delta_g\), and finally
\(\varepsilon_0\). Make the variation bounds in (66) smaller than the score
radius buffers and half of the \(2\beta_*\) score-orientation gap, and require
\[
C\{\delta_g+\varepsilon_0^4\eta\}
 <\operatorname{dist}(K_v^{\rm in},
                       \mathbb R^{15}\setminus V_v^{\rm out}).
\]
In particular, take \(C\eta^2<\beta_*\). A first-exit argument then preserves
\(H\preceq-\beta_* I\), keeps \(\psi\) in \(V_v^{\rm out}\), hence preserves
\(c_g\ge c_v\), and retains all other patch conditions forever. Equations
(65)--(66) imply
\(f\to0,\gamma\to0\) and convergence of all tangential score and value
variables. The limit consequently belongs to \(\mathcal M\). The tube is
bounded and the vector field is smooth at finite parameters, which proves
global existence. Equation (49) proves (59). \(\square\)

# lemma lem:captured-open-cone

## statement

There are a bounded nonempty ambient-open set
\(\mathcal A\subset\mathbb R^{42}\), a number
\(\varepsilon_*>0\), and \(b=\sup_{\xi\in\mathcal A}\|\xi\|<\infty\) such
that
\[
\varepsilon\mathcal A\subset\mathcal F
\quad\hbox{for every }0<\varepsilon<\varepsilon_*.
\tag{67}
\]
Every point in these sets is regular at initialization, its trajectory is
global, and its asymptotics are
\[
\kappa(t)\longrightarrow((1/2,1/2))_{q\ne d},\qquad
R(t)\longrightarrow\frac14,\qquad
S_{qq}(t)-S_{qd}(t)\longrightarrow0.
\tag{68}
\]
Consequently
\[
\liminf_{r\downarrow0}
\frac{\lambda_\Theta(\mathcal F\cap B_r)}
     {\lambda_\Theta(B_r)}
\ge
\frac{\lambda_\Theta(\mathcal A)}
     {b^{42}\lambda_\Theta(B_1)}
>0.
\tag{69}
\]

## proof

Choose precompact open balls of normalized score factors with
\[
\bar Q\ \hbox{near }I,\qquad
\bar K\ \hbox{near }-I,\qquad
\bar E\ \hbox{near }\alpha I,
\tag{70}
\]
where \(\alpha>0\) and the three radii are small enough that every matrix is
invertible, (53) holds,
\[
\operatorname{sym}(\bar Q^{\top}\bar K)\preceq-4\beta_*I,
\quad
\|\bar Q-I\|+\|\bar K+I\|<\rho_*/2,
\quad
\|\bar E\|<M_*/2,
\quad \|P\bar E\|<\eta/2.
\]
Choose also a precompact
open annular cone of value factors whose directions lie in the set \(V\) from
Lemma \(\mathrm{lem:fast-transfer}\), small enough that its \(g_c\)-hitting
image lies in the interior of \(K_v^{\rm in}\). Their Cartesian product is a
bounded ambient-open set \(\mathcal A\) of positive \(42\)-dimensional
measure.

Take \(g_c=3/2-\delta_g/4\), reducing \(\delta_g\) from Lemma
\(\mathrm{lem:uniform-tube}\) if necessary. Lemma
\(\mathrm{lem:fast-transfer}\) shows, uniformly for
\(\xi\in\mathcal A\), that the trajectory from \(\varepsilon\xi\) first
reaches \(g_c\) with
\[
(E,Q,K)=\varepsilon(\bar E,\bar Q,\bar K)+O(\varepsilon^3),
\qquad S=O(\varepsilon^4).
\]
Thus the normalized score displacement is \(O(\varepsilon^2)\). The strict
initial buffers imply, uniformly on \(\mathcal A\),
\[
\operatorname{sym}\!\left((Q/\varepsilon)^{\top}
                           (K/\varepsilon)\right)
 \preceq-3\beta_*I,
\qquad \|PE/\varepsilon\|<\eta
\]
for every sufficiently small \(\varepsilon\). Its value factors lie in
\(K_v^{\rm in}\), with \(c_g\) bounded below, by the choice of the hitting
image, (42), and Euler's identity. Since
\(g_*(s)=3/2+O(\varepsilon^4)\), this hitting point satisfies (58) and the
gain-neighborhood condition for every sufficiently small \(\varepsilon\).
Lemma \(\mathrm{lem:uniform-tube}\) then gives the global trajectory and
(68). Invertibility before capture follows from Lemma
\(\mathrm{lem:fast-transfer}\), so the initial points are regular. This proves
(67).

For \(r<b\varepsilon_*\), take \(\varepsilon=r/b\). Then
\(\varepsilon\mathcal A\subset\mathcal F\cap B_r\), and scaling Lebesgue
measure gives
\[
\frac{\lambda(\mathcal F\cap B_r)}{\lambda(B_r)}
\ge\frac{\varepsilon^{42}\lambda(\mathcal A)}
         {(b\varepsilon)^{42}\lambda(B_1)}.
\]
This is (69). \(\square\)

# lemma lem:reachable-face-and-access

## statement

The basin in Lemma \(\mathrm{lem:captured-open-cone}\) is a reachable
wrong-boundary component inside \(\mathcal U\). It is a finite
contrast-access singularity, not an instability at infinity. Along every
captured trajectory,
\[
\int_0^\infty\mathsf A_\kappa(\theta_t,e_t)\,dt
=\frac N2\log\frac{R(0)}{1/4}<\infty.
\tag{71}
\]
Thus it is an explicit counterexample certificate for N5, N6, and N7.

## proof

The \(g=1\) section enters \(\mathcal U\) by Lemma
\(\mathrm{lem:entrance-region}\), and that set is forward invariant. The
eventual manifold (48) has \(PE=0\), so the quotient task-contrast gradient is
not transmitted through the repeated embedding factor. It is finite and has
positive gain. Since \(R(t)\to1/4>0\), (16) can be divided by \(R\) and
integrated:
\[
\int_0^T\mathsf A_\kappa(\theta_t,e_t)\,dt
=\frac N2\{\log R(0)-\log R(T)\}.
\]
Letting \(T\to\infty\) gives (71). In particular the integrated access is
finite even though
\(G_S=-P/8\) remains a nonzero quotient contrast signal at the limiting face.
\(\square\)

# lemma lem:counterexample-contract

## statement

The family (67) has the following complete counterexample data.

1. It uses the frozen exact-softmax \(C=d=3,m=2\) MQAR population, its
   half-squared population loss, and the active Euclidean factorization.
2. Its trigger is the open product condition (70), the small contrast
   inequality in (53), and the positive value angular cone \(U_v\) surrounding
   (39).
3. Its mechanism is positive-gain selection followed by attraction to the
   finite access-collapse manifold (48).
4. It has ambient positive measure and the strictly positive lower density
   (69); the limiting manifold itself has codimension seven.
5. Its delivered kernel, risk, margins, and factor direction are (68), with
   \(PE\to0\).
6. Its cumulative access is finite by (71).
7. Under every scaled base law charging a compact subset
   \(A\Subset\mathcal A\), its limiting failure probability is at least
   \(\nu(A)>0\); for normalized Lebesgue measure on \(A\) it is one.
8. The weakest initialization-law exclusion certified by this witness is
   \(\nu(\mathcal C_-)=0\), where \(\mathcal C_-\) is the union of the
   captured normalized patches defined by \(U_v\), (53), and (58).
9. The witness does not claim that the complement of \(\mathcal C_-\)
   succeeds, nor does it classify every other boundary component.

A simpler checkable but stronger exclusion of this particular mechanism is
that the base law give zero mass to simultaneous positive value directions,
negative-definite \(\operatorname{sym}(\bar Q^{\top}\bar K)\), and
sufficiently small normalized contrast embedding \(P\bar E\). No
condition in the finite structured selection definition below excludes the
witness.

## proof

Items 1--7 and 9 are Lemmas \(\mathrm{lem:population-algebra}\),
\(\mathrm{lem:value-clock}\), \(\mathrm{lem:uniform-tube}\),
\(\mathrm{lem:captured-open-cone}\), and
\(\mathrm{lem:reachable-face-and-access}\). By definition,
\(\mathcal C_-\) is exactly the union of direction patches for which the
proved capture argument applies. Hence assigning it zero base-law mass is the
weakest law-level condition that excludes exactly this certified mechanism;
the stated sign/access condition is an immediately checkable sufficient way
to do so.

For the N9 check, define a finite structured selection population explicitly:
a structural state \(\omega\) ranges over a finite set; it specifies
\(m_\omega\ge2\) memory values and a target index \(J(\omega)\) from
query/key structure alone; conditional on \(\omega\), the values
\(v_i\in\mathbb R^r\) are independent and centered, are independent of the
target choice, and have common covariance \(\Sigma_v\succ0\); and the label is
\(Y=A_*v_{J(\omega)}\).  For a value-linear model
\[
f_\theta(\omega,v)=\sum_{i=1}^{m_\omega}
\mathcal K_{\omega i}(\theta)v_i,
\]
independence gives the exact identity
\[
R(\theta)=\frac12\mathbb E_\omega\sum_i
\left\|(\mathcal K_{\omega i}(\theta)
-A_*\mathbf1\{i=J(\omega)\})\Sigma_v^{1/2}\right\|_F^2.
\tag{72}
\]

The frozen population of Definition \(\mathrm{def:frozen-mqar}\) belongs to
this class by taking
\(\omega=(q,d,\text{slot ordering})\), \(m_\omega=2\), \(r=1\),
\(J(\omega)\) equal to the slot carrying key \(q\), independent Rademacher
values, \(\Sigma_v=1\), and \(A_*=1\).  All six \(q\ne d\) occur uniformly,
so every target occurs twice and target coverage is complete.  Equation (72)
then reduces to (5), proving kernel identifiability because every summand is
nonnegative and \(\Sigma_v=1>0\).

The target kernel is representable in the closure of the exact-softmax model:
take \(E=Q=I\), \(K=LI\), and \(w=z=e_1,O=I\), so \(g=1\).
Then \(S=LI\), and for every
\(q\ne d\),
\[
a_{qd}=\frac{e^L}{e^L+2}\longrightarrow1,\qquad
b_{qd}=\frac1{e^L+2}\longrightarrow0.
\]
The initial patches in (70) also have \(E,Q,K\) invertible and
\(c_g>0\), so ordinary finite factor access is present; what fails is a
uniformly retained contrast-access bound.  Finally, uniform weighting of all
ordered pairs and the complete Rademacher cube supply exactly the
target--distractor permutation symmetry used in (29), (35), and (49).

Thus the frozen instance has kernel identifiability, target coverage, positive
value covariance, representability, initial factor access, and the invoked
symmetry, yet it has the positive-density basin above.  These properties alone
cannot imply a small-initialization success theorem for the whole finite
structured selection class.  Any valid lift needs an additional
initialization/factor-access condition excluding \(\mathcal C_-\). This is the
promised N9 counterexample. \(\square\)

# theorem thm:T-small

## statement

Work in the active Euclidean parameter space
\(\Theta\cong\mathbb R^{42}\). Let \(\phi_t(\theta_0)\) denote the maximal
gradient-flow solution and define the failure set
\[
\mathcal F
=
\left\{
\theta_0:
\phi_t(\theta_0)\text{ is not global}
\quad\text{or}\quad
\limsup_{t\to\infty}R(\phi_t(\theta_0))>0
\right\}.
\tag{20}
\]

The primary prove-or-refute target is the density theorem
\[
\boxed{
\lim_{r\downarrow0}
\frac{\lambda_\Theta(\mathcal F\cap B_r)}
     {\lambda_\Theta(B_r)}=0.
}
\tag{T_{\rm small}}
\]

Equivalently, for every fixed base law \(\nu\ll\lambda_\Theta\) and
\[
\theta_0^\varepsilon=\varepsilon\xi,
\qquad
\xi\sim\nu,
\]
the failure probability tends to zero as \(\varepsilon\downarrow0\). Fixed
invertible block scalings, including fixed fan-in constants, can be absorbed
into \(\nu\) and preserve the density-zero question.

For any fixed \(\varepsilon>0\), exact probability one is not claimed: a
full-support law still assigns positive, possibly extremely small, mass to the
distant open horn.

The stronger raywise statement
\[
\text{for a.e. }\xi\text{, there exists }\varepsilon_0(\xi)>0
\text{ such that every }0<\varepsilon<\varepsilon_0(\xi)\text{ succeeds}
\tag{T_{\rm ray}}
\]
is optional. It implies \(T_{\rm small}\) but is not equivalent to marginal
failure probability tending to zero.

Neither theorem may assume \(g_0\ge0\), exact balance, \(K=Q\), aligned
attention, bounded factors, bounded \(S\), or a uniform access constant.

## proof

The boxed statement is false. Lemma
\(\mathrm{lem:captured-open-cone}\) constructs an ambient-open bounded set
\(\mathcal A\) of rescaled directions for which
\(\varepsilon\mathcal A\subset\mathcal F\) at every sufficiently small scale.
In fact (69) gives
\[
\liminf_{r\downarrow0}
\frac{\lambda_\Theta(\mathcal F\cap B_r)}
     {\lambda_\Theta(B_r)}>0,
\]
which is stronger than the positive-upper-density requirement for a
refutation.

Every trajectory in this family is global but converges to the wrong delivered
kernel:
\[
x_{qd},y_{qd}\longrightarrow\frac12,\qquad
R\longrightarrow\frac14,\qquad
S_{qq}-S_{qd}\longrightarrow0.
\]
Its cumulative task-kernel access is finite by (71). The failure is therefore
not the predecessor's distant negative-gain horn; it is a finite,
positive-gain, small-initialization access-collapse basin inside
\(\mathcal U\).

By Lemma \(\mathrm{lem:quantifiers}\), normalized Lebesgue measure on
\(\mathcal A\) is an absolutely continuous base law whose failure probability
is one for every sufficiently small scale. More generally, any base law
charging a compact captured patch has failure probability bounded away from
zero. The same open set also disproves \(T_{\rm ray}\).

The exact missing-condition witness and the strongest conclusion supported
after refutation are recorded in Lemma
\(\mathrm{lem:counterexample-contract}\): a corrected initialization theorem
must at least give zero mass to the captured negative-orientation,
low-contrast-access directions \(\mathcal C_-\). This exclusion is necessary
for removing this mechanism only; no sufficiency claim on its complement is
made. \(\square\)
