# Pythia-70M float64 校准：阶段判断

审计：8/8 checkpoints、32 checkpoint×template rows 全部闭合；P10→P11 逐 slot 重构通过；最大 parallel-residual 误差 `2.043e-14`（阈值 `1e-5`）。统计单位只有 **1 条 pretraining trajectory**。

定义：`f=tanh((log p(plus)-log p(minus))/2)`，`R=E[(f-y)^2]/2`，`S_key=E[δ_target-mean(δ_distractor)]`，其中 `δ_i=y(f-f^(-i))` 只阻断 final-prompt receiver→full-card 直接边。

观察：最高 accuracy `0.715`（step4000/bracket_dictionary），最低 risk `0.379`；`S_key` 范围 `-5.878e-05` 到 `2.230e-01`。最强 observation-only head selectivity `1.840e-01`（step143000/line_records/L5H4）；最大自然 swap MSE `3.578e-02`。final：compact_cards: acc=0.535, R=0.510, L_W=0.163, S_key=0.0402; line_records: acc=0.531, R=0.507, L_W=0.147, S_key=0.0226; prose_facts: acc=0.543, R=0.503, L_W=0.074, S_key=-5.88e-05; bracket_dictionary: acc=0.562, R=0.486, L_W=0.136, S_key=0.0569。

判断：`diffuse → selective routing → sparse collision/downstream reorganization` **不成立为当前结论**。最终四模板未通过描述性稳定 retrieval screen；且未保存逐 episode 自然-swap 差值，不能检验 collision 稀疏/重尾。三类 finite patch 只能说明 nonlinear suffix 对 swap 有位置依赖重组，不能唯一归因于 QK、OV 或 FFN。

边界：checkpoint/template/layer/head 均是 repeated measures；无 seed-level 推断。P10 不是 total mediation。Pythia 只验证 instrumentation 与弱、模板异质的 routing 信号；toy 的多-seed rank/collision 结果不能据此外推为 GPT 训练定律。

结论：完整故事支持=`false`。论文主理论仍应留在 toy 可识别问题：在 learned compressed `E` 与 per-head rank 下，何种 margin/cover 条件使低风险强迫 `S_key>0`，以及何时存在低风险但 `S_key≤0` 的反例。
