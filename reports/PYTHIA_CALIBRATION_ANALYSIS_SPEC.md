# Pythia-70M 训练轨迹校准：冻结分析规格

**状态：** instrumentation calibration；只允许描述性结论，不进入论文确认性统计。  
**冻结时间：** 2026-08-20，在 8-checkpoint v3 校准完成并读取最终指标之前。  
**模型族：** `EleutherAI/pythia-70m` 的单条公开训练轨迹。checkpoint、template、
skeleton、value assignment 都不是独立训练 seed。

---

## 1. 校准回答什么，不回答什么

这次运行有三个目标：

1. 验证真实 GPT-NeoX tokenizer、parallel residual、attention mask、Q/K/V capture 和三类
   finite patch 在全部 checkpoint 上可运行且精确闭合；
2. 估计 8-checkpoint 全测量的 wall time、GPU memory、磁盘体积和失败恢复行为；
3. 检查冻结 prompt population 是否在某些训练阶段产生可测 retrieval 信号，以决定后续
   多训练 seed study 是否有统计功效。

它不能证明：

- 训练“导致”某一机制；这里仅有一条训练轨迹；
- 某个 template 的差异可推广到自然语言；
- direct-edge block 是 total mediation；
- 某层或某模块是 compensator；
- 32 个 checkpoint-template rows 是 N=32。

后续训练动力学推断的独立单位必须是不同 pretraining seeds，优先使用 PolyPythias 的
冻结 seed family；checkpoint 是 seed 内 repeated measure。

---

## 2. 冻结设计

训练 revisions 为

\[
\mathcal T=\{0,64,512,1000,4000,16000,64000,143000\}.
\tag{PC1}
\]

每个 revision 使用 4 个预先冻结的 prompt templates；每个 template 有 16 个 concept
skeletons。每个 skeleton 枚举完整的 4-bit Rademacher value cube：

\[
v=(v_1,v_2,v_3,v_4)\in\{-1,+1\}^4.
\tag{PC2}
\]

因此每个 template 有 256 prompts，每个 revision 有 1024 prompts，完整 calibration 共
8192 prompt cases。所有 templates 使用同一套抽象 skeleton/value assignments；tokenizer
层面另有严格的 contextual span-alignment audit。

memory 中的可见 values 是 `plus/minus`，teacher-forced answer suffix 是带前导空格的
` plus/ minus`。二者被显式分开，避免 card format 产生双空格或错误 BPE boundary。

---

## 3. 输出分数、风险和完整 Walsh 分解

对 prompt \(x\)，模型对两个完整 answer suffix 的条件 log likelihood 分别为
\(\ell_+(x),\ell_-(x)\)。统一的有界输出是

\[
f_t(x)=\tanh\!\left(\frac{\ell_+(x)-\ell_-(x)}2\right)\in[-1,1].
\tag{PC3}
\]

label \(y\in\{-1,+1\}\)，风险为

\[
R_{t,k}=\frac12\,\mathbb E_{x\in\Omega_k}[f_t(x)-y(x)]^2,
\tag{PC4}
\]

其中 \(k\) 是 template，\(\Omega_k\) 是该 template 的冻结有限 population。

对每个固定 skeleton，完整 Boolean cube 给出唯一 Walsh 展开

\[
f_t(v)=\sum_{A\subseteq[4]}\widehat f_{t,A}\chi_A(v).
\tag{PC5}
\]

必须分别报告：

- \(E_T\)：target linear coefficient 对 1 的误差；
- \(L_D\)：distractor linear coefficients；
- \(L_H\)：higher-order interactions；
- \(L_0\)：bias；
- \(L_W=L_D+L_H+L_0\)：总非 target leakage；
- direct half-MSE 与 Parseval partition 的相对误差。

accuracy 只是符号正确率，不能替代风险或 Walsh leakage。

---

## 4. 三种不同的“干预”不得混名

### 4.1 自然 on-support swap

只把一个非 target concept 替换成当前 prompt 未出现的合法 concept；values、target、
label、slot 和模板不变：

\[
I_{swap,t,k}=\mathbb E[f_t(X^{swap})-f_t(X)]^2.
\tag{PC6}
\]

它测函数 cross-talk，但本 calibration 的 16 skeletons 很小，只作描述。

### 4.2 注册 direct-edge effect

对 episode \(e\) 和 memory slot \(i\)，在每一层、每一 head 的 softmax 前，将最后一个
prompt decision receiver 到该完整 value-bearing memory-card span 的 score 置为
\(-\infty\)，并重新计算完整 answer likelihood。记 masked score 为
\(f_t^{(-i)}(X_e)\)，则

\[
\delta_{e,i}=y_e\{f_t(X_e)-f_t^{(-i)}(X_e)\},
\tag{PC7}
\]

\[
S_{key,t,k}
=\mathbb E_e\left[
\delta_{e,J_e}
-\frac1{m-1}\sum_{i\ne J_e}\delta_{e,i}
\right].
\tag{PC8}
\]

这是一个**直接的、final-prompt receiver 到 full-card 的路径 effect**。later answer tokens
仍可读取 memory，模型也可走间接路径，因此它不是 total memory attribution。

### 4.3 三类 finite activation patch

必须分开画、分开命名：

1. `source_span_transmission`：changed concept source span 的传输；
2. `decision_receiver_accumulation`：query/decision receiver 已积累的 cross-talk；
3. `coherent_replay_gate`：替换 residual input 后让 attention 和 FFN 两个 parallel
   branches 一起重算，用作真实 suffix replay gate。

每一项报告

\[
P_{role,site}=\mathbb E[f^{patch}_{role,site}(X)-f(X)]^2.
\tag{PC9}
\]

这些是重叠的 local hybrid estimands，不构成跨模块加法分解。相邻 coherent sites 的等价
只验证 instrumentation，不能制造 attenuation；没有共同 base factorial/Shapley closure
时，不得写“cross-talk 在模块 M 被消除”。

---

## 5. attention/head diagnostics 的正确粒度

每个 episode、layer、head、memory slot 保存：

- decision receiver 到 concept span 和完整 card span 的 attention mass；
- post-RoPE key RMS、value RMS、query norm；
- per-head pre-OV receiver norm。

用 `episode_index/template_id/slot` 和 direct-edge sidecar 中的 `target_slot` 精确 join，
得到

\[
A^{sel}_{t,k,\ell,h}
=\mathbb E\left[
a_{\ell h,J}
-\frac1{m-1}\sum_{i\ne J}a_{\ell h,i}
\right].
\tag{PC10}
\]

它是 observation-only attention selectivity，不是 \(S_{key}\)。图中必须同时显示
target 和 distractor mass，不能只显示差值；layer/head/episode 也不能当独立 N。

---

## 6. 完成与完整性 gate

校准只有在以下条件全部满足时才算完成：

1. 8/8 revision directories 有原子 `_SUCCESS`，root `_SUCCESS` 存在；
2. `failures.jsonl` 没有 unresolved revision；
3. 正好 32 个 revision-template wide rows；
4. 每 revision 正好：
   - 4096 direct-edge episode×slot rows；
   - 196608 head-diagnostic episode×layer×head×slot rows；
   - 24576 finite-patch episode×layer×site rows；
   - 6144 parallel-residual episode×layer rows；
5. 单一 execution-environment identity，且 deterministic/TF32/cuDNN flags 完整；
6. 所有 revisions 的 schema、measurement-contract/source hashes 和 prompt population hash
   一致；
7. raw NPZ 的 SHA-256、aggregate JSON/CSV、checkpoint rows 和 root aggregates 可以严格
   互相重构；
8. parallel residual identity

   \[
   \Delta h_{post}=\Delta h+\Delta h_{attn}+\Delta h_{ffn}
   \tag{PC11}
   \]

   的 maximum absolute closure error 不超过 \(10^{-5}\)；
9. 模型 weights、grads、hooks、mode 和 RNG 在 observation-only measurement 后不变。

任何 gate 失败都先修 instrumentation；不得把失败 checkpoint 从轨迹中删除后继续解释。

---

## 7. 冻结图表

所有图都以 revision 为 x 轴，4 templates 全部显示；不画伪造的 seed confidence interval。

1. **功能轨迹：** accuracy、risk、value-flip effect；
2. **Walsh 轨迹：** \(E_T,L_D,L_H,L_0,L_W\)，使用共享纵轴或明确标注 log scale；
3. **自然/直接因果：** \(I_{swap}\)、target-edge、mean-distractor-edge、\(S_{key}\)；
4. **层头 routing：** target/distractor full-card attention mass 和 \(A^{sel}\) 的
   checkpoint×layer×head heatmaps；
5. **三类 patch curves：** role 分面、site/layer 为 x 轴；绝不把三个 role 合成一条
   “compensation curve”；
6. **parallel residual audit：** branch chord norms 与 closure error；
7. **template robustness：** 每个 metric 的四条 template trajectories；只描述异质性，
   不事后选择表现最好的 template。

图例必须写明：one pretraining trajectory、checkpoint is repeated measure、calibration
only。

---

## 8. 发布范围

本地保留完整 raw CSV 方便逐行审计；GitHub 主仓优先发布：

- config、代码、测试和 prompt audit；
- 每 revision 的压缩 numeric NPZ 与 metadata/hashes；
- root wide/tidy aggregates；
- 派生统计表、PNG+SVG 和中文报告；
- 精确重建命令和环境锁。

体积巨大的 CSV 是 NPZ 的冗余文本表示，可在 release artifact 或生成脚本中提供，不应仅为
“看起来完整”而让主仓膨胀数百 MB。任何排除都必须在 public manifest 中显式列出，而不是
静默删除 provenance。

---

## 9. 校准后的决策规则

- 若 8 revisions 全部接近 chance，结论是当前 synthetic prompt 没有足够信号；保留
  instrumentation 结果，但不启动 512-skeleton production。
- 若出现稳定 retrieval 和可测 causal signal，先验证严格 reader、runtime/storage 和图表，
  然后用同一冻结四模板启动更大 prompt population；不得根据 calibration 选择 template。
- 单条 deduplicated Pythia trajectory 永远只作描述。论文级训练效应须在独立
  pretraining-seed family 上复验，并以 seed-block inference 处理 checkpoint trajectory。
- 只有 toy rank/collision estimands 与真实 checkpoints 呈同方向、且跨训练 seeds 复制后，
  才讨论 toy-to-GPT mechanism transfer。
