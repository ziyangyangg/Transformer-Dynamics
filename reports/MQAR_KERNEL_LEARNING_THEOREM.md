# MQAR Kernel Learning Under Population Gradient Flow

## Status

The unrestricted statement is false. A precise positive theorem holds for a
one-layer, one-head, value-linear, permutation-symmetric exact-softmax parameterization with
nondegenerate positive factor initialization. Two obstructions are exact:

1. zero query and key factors form a routing-learning barrier even when the
   composite score gradient is nonzero;
2. signed multi-head value gains can achieve zero risk without identifiable
   direct-edge routing.

This is the first condition-discovery slice of the larger program. It proves one
sufficient set of assumptions and shows that factor access and route identifiability
cannot be omitted without replacement. It does not show that the same conditions are
sufficient for an arbitrary learned dictionary or a general residual Transformer.

## 1. Population and reduced Transformer

Let $m\ge 2$. Each episode contains $m$ distinct key-value memories, with independent
values $v_i\sim\operatorname{Unif}\{-1,+1\}$. A target index
$J\sim\operatorname{Unif}[m]$ determines the query key and label $Y=v_J$.

This is the single-query, binary-value specialization of the public MQAR construction:
keys are distinct, the query repeats one key, and the label is its associated value. It
is not the full Zoology sequence generator, which permits multiple queries, token-valued
answers, filler tokens, and a distance law.

Consider one exact-softmax head. The target score exceeds every distractor score and
the zero-value query-self score by $\delta$. Therefore

$$
a(\delta)=\frac{e^\delta}{e^\delta+m},
\qquad
b(\delta)=\frac{1}{e^\delta+m},
\qquad
a+mb=1.
\tag{1}
$$

Here $a$ is the target weight and $b$ is the weight of each of the $m-1$ distractors
and of query-self. Query-self has zero value. With effective value gain $g$,

$$
f(v)=g\left(a v_J+b\sum_{i\ne J}v_i\right).
\tag{2}
$$

Rademacher orthogonality gives the exact population risk

$$
R_m(g,\delta)
=
\frac12\left[(ga-1)^2+(m-1)(gb)^2\right].
\tag{3}
$$

The Transformer factors are retained. Let $q,k,\rho,o,v,w>0$, where $\rho$ is a
learned radial scale of the role-separated concept dictionary. Define

$$
\delta=qk\rho^2,
\qquad
g=ovw.
\tag{4}
$$

This learns the radial dictionary scale, $Q/K$, $O/V$, and the readout. The concept
directions are fixed by the registered role-tied parameterization; arbitrary embedding-direction learning
is outside the theorem.

## 2. Exact closed dynamics

Write $n=m-1$ and define

$$
S(\delta)=a^2+n b^2,
\qquad
D(\delta)=a-\frac{n}{m}b,
\qquad
h(g,\delta)=gD(\delta).
\tag{5}
$$

Direct differentiation of (3) gives

$$
\partial_g R_m=gS-a,
\qquad
\partial_\delta R_m=mgab(h-1).
\tag{6}
$$

Population gradient flow of all six factors is

$$
\begin{aligned}
\dot q&=-k\rho^2\,\partial_\delta R_m,
&
\dot k&=-q\rho^2\,\partial_\delta R_m,
&
\dot\rho&=-2qk\rho\,\partial_\delta R_m,\\
\dot o&=-vw\,\partial_g R_m,
&
\dot v&=-ow\,\partial_g R_m,
&
\dot w&=-ov\,\partial_g R_m.
\end{aligned}
\tag{7}
$$

Consequently, the composite dynamics are preconditioned gradient flow:

$$
\dot\delta=-P_\delta\,\partial_\delta R_m,
\qquad
P_\delta=\rho^4(q^2+k^2)+4q^2k^2\rho^2>0,
\tag{8}
$$

$$
\dot g=-P_g\,\partial_g R_m,
\qquad
P_g=v^2w^2+o^2w^2+o^2v^2>0.
\tag{9}
$$

The source code verifies (3), (6), and (7) against complete value enumeration and
automatic differentiation.

## 3. Kernel-learning theorem

**Theorem 1.** Assume $m\ge2$, all six initial factors are positive, and

$$
h(g_0,\delta_0)<1.
\tag{10}
$$

Under (7),

$$
\delta(s)\longrightarrow+\infty,
\qquad
g(s)\longrightarrow1,
\qquad
R_m(s)\longrightarrow0.
\tag{11}
$$

Moreover, the learned interaction-kernel transport error is

$$
\mathcal E_{\mathcal K}(s)
=
(ga-1)^2+(m-1)(gb)^2
=
2R_m(s)
\longrightarrow0.
\tag{12}
$$

Thus training makes the target score dominate every distractor and query-self score,
while the $OV$/readout gain converges to the correct value transport.

**Proof.** Positivity and (10) imply $\partial_\delta R_m<0$. At the boundary $h=1$,
equations (6), (8), and (9) give $\dot\delta=0$ and

$$
\dot h
=
P_g(aD-S)
=
-\frac{m-1}{m}P_g b
<0.
\tag{13}
$$

Hence $h<1$ is forward invariant and $\dot\delta>0$.

For $\delta\ge0$, $S<a$. Therefore $\dot g>0$ whenever $g\le1$, and

$$
g(s)\ge\min\{g_0,1\}>0.
\tag{14}
$$

Also $S\ge a^2$ and $a\ge1/(m+1)$, so $\dot g<0$ for $g>m+1$. Hence $g$ is bounded.

The factor flow preserves

$$
q^2-k^2,\qquad
\rho^2-2q^2,\qquad
o^2-v^2,\qquad
o^2-w^2.
\tag{15}
$$

If $\delta$ were bounded, (14), (15), and the boundedness of $g$ would place every
factor in a compact positive set. Along the flow,

$$
\dot R_m
=
-P_\delta(\partial_\delta R_m)^2
-P_g(\partial_g R_m)^2
\le0.
\tag{16}
$$

LaSalle's invariance principle would then require a limit point satisfying both
partial derivatives in (6) equal to zero. The first equality implies $h=1$, but at
$g=1/D$,

$$
\partial_g R_m
=
\frac{S-aD}{D}
=
\frac{(m-1)b}{mD}
>0,
\tag{17}
$$

a contradiction. Therefore $\delta(s)\to+\infty$.

Now $a\to1$, $b\to0$, $S\to1$, and $a/S\to1$. Equation (9) can be written as

$$
\dot g=P_gS\left(\frac{a}{S}-g\right).
\tag{18}
$$

The positive lower bound on $g$ and the invariants (15) keep $P_gS$ bounded away from
zero. Equation (18) traps $g$ in every neighborhood of $1$, so $g\to1$. Substitution
into (3) proves (11), and (12) follows directly. $\square$

## 4. Exact refutation of the unrestricted statement

Set $q_0=k_0=0$, with $\rho_0>0$ and a positive value path satisfying $h<1$. Then

$$
\partial_\delta R_m<0,
\qquad
\dot q=\dot k=\dot\rho=0.
\tag{19}
$$

The direct composite margin would move, but the factorized $Q/K$ system cannot leave
this invariant set. Value-path training converges to the best uniform-attention gain,
and the limiting risk is

$$
\inf_g R_m(g,0)=\frac{m-1}{2m}>0.
\tag{20}
$$

Therefore no theorem valid for every initialization can claim that factorized
population gradient flow learns the routing kernel.

A separate two-head exact-softmax construction in
[CAUSAL_ROUTING_BRIDGE_THEOREM.md](CAUSAL_ROUTING_BRIDGE_THEOREM.md) has
$R=0$ and correct value dependence but $S_{\rm key}=0$ because signed head gains
cancel under slot blocking. Thus low risk alone does not identify the internal direct
route.

## 5. Precise boundary and next object

Established:

$$
\text{positive nondegenerate factors}
+\text{one identifiable head}
+\text{symmetric radial dictionary}
\Longrightarrow
\text{correct margin and value transport}.
\tag{21}
$$

Not established: arbitrary learned embedding directions, RMSNorm, multiple heads,
residual bypasses, FFNs, or multi-layer composition. These are not silently covered by
Theorem 1.

Gate 1 is therefore resolved as follows: the unrestricted kernel-learning claim is
false, while a nontrivial exact factorized reduced model admits a complete positive theorem.
The next data object is LEGO, using the published state-tracking law and the same
Transformer family. The next theorem must replace the single required edge by the two
known inputs of each transition and must not repeat the existing LEGO learnability
claim.
