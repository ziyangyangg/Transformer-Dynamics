# 下一轮研究任务

本文件只列当前证据真正支持的下一步。优先级按“先补缺失 estimand，再增加训练量，最后
尝试 theorem”排序；任何新现象仍须先做文献查重、已知优化 remedy 和负对照。

## P0-1：真正评估注册的 causal key selectivity

当前只测了 target edge：

\[
\delta_J(U)=Y\{f(X)-f(\operatorname{do}(s_{TJ}^{\ell h}=-\infty,\forall \ell,h))\}.
\]

下一步必须对每个 episode 的每个 memory slot 都重算一次：

\[
S_{\rm key}=\mathbb E_U\left[
\delta_J(U)-\frac{1}{m-1}\sum_{i\ne J}\delta_i(U)
\right].
\]

实现任务：

1. evaluator 输出 `[batch, memory_slot]` blocked-edge effects，并验证 causal mask、重归一化和
   descendants 全部重算；
2. 用手算单层模型测试 target/distractor effects，而不是用 attention mass 代替；
3. 对已保存 snapshots 用相同 held-out streams 重放，seed 内先平均 episode/slot，再做
   whole-seed bootstrap；
4. 报告 \(\mathbb E\delta_J\)、distractor mean 和二者差，三者不可合并成一个无来源分数；
5. 在这一步完成前，不使用 “direct-key selective routing 已通过” 的表述。

## P0-2：实现预注册的非对称 QK finite estimand

当前 midpoint split 把双线性交互平均分给 content/route，不能检验预注册命题。下一步对
同一个 on-support chord 保存：

\[
\Delta f=C_0+R_0+I,
\]

其中 \(C_0\) 固定 base routing 只换 content，\(R_0\) 固定 base content 只换 routing，
\(I\) 是剩余 finite interaction。必须同时报告：

\[
C_{QK}=\mathbb E_{\rm chord}\left[
\log\frac{(C_0+I)^2+\varepsilon}{(C_0+R_0+I)^2+\varepsilon}
\right].
\]

以及 \(C_0,R_0,I\) 的 signed/energy 版本。实现后先用解析 softmax 小例子证明它与 midpoint
split 可反号，再重放现有 snapshots。只有这个量通过预注册方向、finite output gate 和优化器
复制，才能检验 QK suppression 命题。

## P0-3：用新 seeds 做 remedy 确认实验

现有 b=2,048 follow-up 复用了筛选后的 cells/seeds，所以只能探索。下一轮先冻结协议，再运行：

- 难 cell：d=8, C=32, m=4, H=4，分别 no-FFN 与 FFN16；
- 训练：LR=.003，steps 800/1600/3200/6400；constant LR 与 cosine decay；
- controls：一个已稳定 H=1 cell、一个 load=1 H=4 cell；
- discovery-confirmation 分离：remedy 选择使用一组新 seeds，最终结论再用从未调参的第二组
  seeds；
- 固定 b=2,048 evaluation stream，主要 endpoint 为 swap MSE、Walsh leakage、base/donor risk；
- 对 cell × schedule × endpoint family 做 max-T 或预注册的层级校正。

目标不是证明“小学习率更好”，而是区分：未收敛、优化路径依赖、head bottleneck 与稳定容量
残差。

## P1-1：把补偿定位到有限模块作用

对每个候选模块/site \(M\) 和 episode \(e\)，令 \(z_{M,e}\) 是该 site 的 base state，
\(G_{M,e}\) 是从该 state 到最终 prediction 的固定 nonlinear suffix。先定义真正的 finite
response，而不是 Jacobian gain：

\[
p_{M,e}(\Delta)=G_{M,e}(z_{M,e}+\Delta)-G_{M,e}(z_{M,e}).
\]

在同一 site 分别构造 on-manifold distractor chord \(\Delta^d_{M,e}\) 与 target-value chord
\(\Delta^t_{M,e}\)，并比较 finite gains

\[
g_{M,d}^2=
\frac{\mathbb E_e[p_{M,e}(\Delta^d_{M,e})^2]}
{\mathbb E_e\|\Delta^d_{M,e}\|_2^2+\varepsilon},
\qquad
g_{M,t}^2=
\frac{\mathbb E_e[p_{M,e}(\Delta^t_{M,e})^2]}
{\mathbb E_e\|\Delta^t_{M,e}\|_2^2+\varepsilon}.
\]

只有在两个输入能量均超过预注册 practical floor 时，才检验
\(g_{M,d}\le\rho<\gamma\le g_{M,t}\)。QK、OV、FFN、readout 位于不同 site，必须各自记录
\(z_{M,e}\)、suffix 和 residual scaling，不能直接比较未归一化的 \(\Delta\)。

候选模块必须同时通过 target preservation、distractor attenuation、whole-seed replication 和
functional equivalence；相邻确定性节点的 coherent donor patch 不作为补偿证据。若多个模块的
interaction 不可忽略，则目标转为证明“补偿不可唯一模块化”的 counterexample。

## P1-2：闭合 learned-E composite routing 的 population GF

从单层、单头、无 FFN、value-blind score 的可枚举模型开始。追踪：

- embedding Gram \(G=EE^\top\)；
- target/distractor score moments；
- value-readout overlap；
- Q/K 与 O/V factor imbalance；
- risk 与 Walsh target/distractor energy。

第一个 theorem/anti-theorem 目标是：exchangeable state 在何种初始化/温度/负载下失稳并选择
target routing；若不能唯一选择，则构造两个同低风险但内部 routing 不同的稳定 attractors。
之后才增加多头、FFN 与有限样本 SGD。

## P1-3：重新注册 architecture/rank family

当前 7 个 normalized-rank contrasts 是选择后的未校正 exploratory results。可以先对完整
secondary family 补做 BH q=.10，但结果仍只能叫 multiplicity-adjusted exploratory。若要称为
confirmatory，必须先冻结完整 factorial family、方向和决策规则，再用未参与选择的新 seeds/data
复验。实验设计还必须把以下三种缩放分开：固定 C 改 d、固定 C/d 同时改 C,d、固定总参数量
改 heads/d_head。

## 暂停条件

只有当以上已知 estimand、优化 remedy、独立 seeds、multiplicity 和 finite module controls 都
不能解释稳定现象时，才把它升级为新的 open problem。下一轮最先开工的是 P0-1 和 P0-2；
它们直接修复当前报告中最重要的两个“尚未真正测到”的理论量。
