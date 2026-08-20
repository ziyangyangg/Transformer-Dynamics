# Current checklist

## Theory first

- [ ] 冻结一个具有已知 \(G^*(X)\) 的公开生成任务及 population law。
- [ ] 写出完整 \(E,Q,K,V,O,w\) gradient-flow 方程。
- [ ] 推导 \(B=Q^\top K\)、\(C=OV\) 与 margin/transport order parameters。
- [ ] 证明 order-parameter closure，或给出不闭合反例。
- [ ] 证明 kernel-alignment 定理，或构造完整 exact-softmax 反例。

## Training-to-depth bridge

- [ ] 从 learned score margin 推出 softmax leakage 界。
- [ ] 从 learned \(C\) 推出 value-transport 误差界。
- [ ] 把单层误差传播到有限深度任务误差。
- [ ] 明确 training time 与 depth time，不混用两类极限。

## Evidence

- [ ] 用已有 toy 轨迹重算上述理论变量；不新增无关指标。
- [ ] 在新独立 seeds 上验证定理方向与有限宽度误差。
- [ ] 只有 Gate 1/2 通过后，运行公开任务与 20M–70M 多-seed 验证。
- [ ] 将 Pythia 仅作为 single-trajectory descriptive boundary。

## Publication

- [ ] 每项主张标注 theorem / counterexample / multi-seed evidence / descriptive。
- [ ] 文献表逐条说明已解决内容、明确缺口和本工作新增内容。
- [ ] README、计划、报告与图表只使用研究宪章中的统一对象。
