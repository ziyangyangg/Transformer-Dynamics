# Autoresearch classic loop

Mode: classic
Iterations: 15

Metric: evidence completeness, measured by the count of validated intervention
contracts (support-preserving swap, exact patch replay, JVP/finite-difference tangent,
seed-level serialization, optimizer/architecture replication).  The sign or size of a
scientific effect is deliberately not an optimization metric.

Verify:

```
PYTHONPATH=src /home/zion/miniforge3/envs/llm4rec/bin/python3.11 \
  -m unittest discover -s tests -v
```

Keep an iteration only if it adds a registered contract or improves numerical agreement
without changing an estimand after seeing the result.  Preserve negative findings and
all failed configurations in the ledger.
