# 既有实验在主问题中的位置

主问题只有一条链：

\[
(\mathcal D,R,\theta_0)
\xrightarrow{\mathrm{gradient\ flow}}
\mathcal K_{\theta_s}
\xrightarrow{\mathrm{depth}}
\Phi_{\theta_s}^{L}(X).
\]

下面按是否推进这条链分类。实验规模、代码量和运行成本不决定其理论地位。

## 核心基础：保留并直接复用

| 内容 | 已经建立 | 下一步用途 |
|---|---|---|
| exact-softmax toy model、population law、训练与 checkpoint | 可重复观察 \(E,Q,K,V,O,w\) 的训练轨迹 | 推导并核对 \(B_s,C_s\) 的 gradient-flow 方程 |
| \(B=Q^\top K,\ C=OV\) composites 与完整 trace | 避免 raw-factor gauge 误读，直接表示 interaction law | 定义 kernel order parameters |
| Walsh/value-flip/逐 slot blocking | 区分输出正确与是否使用正确 source | 作为 identifiability 检验，不作为主问题 |
| fixed-kernel clustering baseline | global collapse 可与均匀 attention 同时出现 | 证明 clustering 不能替代 task alignment |
| causal-routing bridge 定理与 signed-gain 反例 | 可识别子类中低风险推出正确 blocking effect；一般模型中该命题无条件为假 | 固定主定理必须包含的 no-bypass/gain/cancellation 条件 |

这些代码和数学结果是当前项目真正的起点。

## 有用的次级证据：保留，但不主导论文

| 实验 | 准确结论 | 地位 |
|---|---|---|
| Phase-II 12-seed training limit | constant/cosine 均未形成稳定不可消除 plateau；更长训练仍改善 | 排除“残差已是新 open problem” |
| rank-matched vs dense composites | same-rank direct 没修复，dense upper bound 强烈改善 leakage | rank/function capacity 候选；不能称纯 factorization optimization |
| high-\(N\) on-support swap | 干扰高度重尾，方向性对照稳定，但 all-state precision gate 仍失败 | 采样设计与稀有误差警告；不是 tail-law 定理 |
| composite planes、NTK、gauge controls | 排除 raw-factor flatness 与纯 lazy 描述等普通误读 | 附录诊断；不能决定主机制 |
| representation/coherence statistics | 高 coherence 可与低 leakage 共存 | 排除“非正交必然导致功能错误”这一充分解释 |

只有当这些量进入 kernel-learning 定理的假设或误差项时，才再次运行。

## 外部校准：只证明工具可用

Pythia-70M float64-v4 完成 8/8 checkpoints、32/32 checkpoint-template rows，数值 closure
通过。但它只有一条 pretraining trajectory，最终四模板 accuracy 约 \(0.53\)–\(0.56\)，
direct-edge effect 非单调且模板异质。

因此它只证明：

- exact tokenization、逐 slot edge mask、head observation 与 finite patch 能在真实 GPT-NeoX
  上一致运行；
- attention mass selectivity 不等于输出对该 edge 的实际依赖；
- 当前 calibration 没有支持
  diffuse → selective → sparse collision → downstream reorganization 故事。

它不能证明一般训练规律，也不支持把 Pythia 扩展到更多 sizes 作为当前优先事项。

## 隔离或停止：不进入科学结论

| 内容 | 原因 | 处理 |
|---|---|---|
| localization-v2 的 62/72 complete cases | runner 把 row-wise 联合容差错误接成 max-relative gate；root 无 success marker | 保留失败诊断；禁止 complete-case inference，不为当前主线重跑 |
| QK midpoint suppression 结论 | 与注册 asymmetric content/route/interaction estimand 不同 | 仅保留 protocol-deviation 记录 |
| OV/FFN “compensator”命名 | local hybrid patches 不构成跨模块加法归因，FFN gates 未闭合 | confirmed compensator 仍为 0；停止命名 |
| population-GF P39 closure | P38 数值收敛 gate 在 representation/functional coordinates 失败 | 记为 blocked，不拟合或解释低维 closure |
| 大范围 architecture factorial patterns | 多数是选择后、探索性或改变多个因素的 contrasts | 不作为主结果，不继续扩 grid |
| rare-collision 论文主线 | toy 有重尾线索，但 Pythia 未保存所需 episode-level tail，且没有跨模型确认 | 降为未来可能机制，不再作为标题 |

## 对已有仓库的处理原则

- 不删除历史代码、配置或结果；它们继续提供审计与反例来源。
- README 首先指向研究宪章、本文件和当前计划，而不是历史机制报告。
- 新实验必须直接测量 \(B_s,C_s,\gamma_s,\mathcal E_{\rm transport}\) 或 depth error。
- 与主定理链无关的指标不再新增；失败实验不通过换名字进入主结论。
