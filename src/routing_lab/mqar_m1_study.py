"""Resumable multi-seed study for the registered MQAR M1 boundary bridge.

Every training seed is an independent unit.  Arms reuse the same raw initialization
and counter-addressed online MQAR batches; only the registered Q/K scale differs.
The optional half-step audit reuses each standard-arm minibatch for two half-sized
updates, so optimizer discretization is changed without changing the sampled vector
field.  Checkpoints, populations, layers, heads, and queries remain repeated measures.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .mqar_m1 import (
    M1ModelConfig,
    M1Transformer,
    MQARTokenBatch,
    ZoologyMQARConfig,
    sample_zoology_mqar_batch,
)

SCHEMA_VERSION = "mqar-m1-boundary-study-v1"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _seed_key(*parts: object) -> int:
    message = ":".join(map(str, (SCHEMA_VERSION, *parts))).encode("utf-8")
    return int.from_bytes(sha256(message).digest()[:8], "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class M1ArmConfig:
    """One paired initialization-scale intervention."""

    name: str
    qk_initial_scale: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("arm name must be nonempty")
        if not isfinite(self.qk_initial_scale) or self.qk_initial_scale < 0.0:
            raise ValueError("qk_initial_scale must be finite and nonnegative")


@dataclass(frozen=True)
class M1TrainingConfig:
    """Frozen finite-step optimizer contract for the M1 bridge."""

    optimizer: str
    learning_rate: float
    steps: int
    batch_tokens: int
    checkpoint_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.optimizer.lower() not in {"sgd", "adamw"}:
            raise ValueError("optimizer must be sgd or adamw")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if self.steps < 1 or self.batch_tokens < 1:
            raise ValueError("steps and batch_tokens must be positive")
        if (
            not self.checkpoint_steps
            or self.checkpoint_steps[0] != 0
            or self.checkpoint_steps[-1] != self.steps
            or tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
        ):
            raise ValueError("checkpoint_steps must be unique and span zero to steps")


@dataclass(frozen=True)
class M1StudyConfig:
    """Complete scientific identity of the M1 experiment."""

    study_id: str
    upstream_commit: str
    upstream_source_sha256: str
    model: M1ModelConfig
    train_populations: tuple[ZoologyMQARConfig, ...]
    evaluation_populations: tuple[ZoologyMQARConfig, ...]
    arms: tuple[M1ArmConfig, ...]
    seeds: tuple[int, ...]
    training: M1TrainingConfig
    evaluation_examples: int
    causal_examples: int
    step_halving_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.study_id.strip():
            raise ValueError("study_id must be nonempty")
        if len(self.upstream_commit) != 40 or len(self.upstream_source_sha256) < 12:
            raise ValueError("upstream source identity is incomplete")
        if not self.train_populations or not self.evaluation_populations:
            raise ValueError("training and evaluation populations are required")
        if not self.arms or len({arm.name for arm in self.arms}) != len(self.arms):
            raise ValueError("arms must be nonempty and uniquely named")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")
        if not set(self.step_halving_seeds).issubset(self.seeds):
            raise ValueError("step_halving_seeds must be a subset of seeds")
        if self.step_halving_seeds and self.training.optimizer.lower() != "sgd":
            raise ValueError("step halving is registered only for plain SGD")
        if min(self.evaluation_examples, self.causal_examples) < 1:
            raise ValueError("evaluation sample sizes must be positive")
        all_populations = (*self.train_populations, *self.evaluation_populations)
        if any(item.vocab_size != self.model.vocab_size for item in all_populations):
            raise ValueError("all populations must share the model vocabulary")
        if any(
            item.sequence_length > self.model.max_sequence_length
            for item in all_populations
        ):
            raise ValueError("population exceeds the model context")


@dataclass(frozen=True)
class M1RunSummary:
    study_id: str
    planned_runs: int
    completed_runs: int
    skipped_runs: int


@dataclass(frozen=True)
class M1ValidatedArtifact:
    study_id: str
    seed_runs: int
    metric_rows: int
    routing_rows: int


def _study_hash(config: M1StudyConfig) -> str:
    return _hash_bytes(_canonical_bytes(asdict(config)))


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("mqar_m1.py", "mqar_m1_study.py")
    return {name: _hash_file(root / name) for name in names}


def _population_from_dict(payload: dict[str, Any]) -> ZoologyMQARConfig:
    return ZoologyMQARConfig(**payload)


def load_m1_study_config(path: Path) -> M1StudyConfig:
    """Load a strict JSON config without implicit defaults outside dataclasses."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.pop("schema_version", None) != SCHEMA_VERSION:
        raise ValueError("unsupported M1 study config schema")
    return M1StudyConfig(
        study_id=payload["study_id"],
        upstream_commit=payload["upstream_commit"],
        upstream_source_sha256=payload["upstream_source_sha256"],
        model=M1ModelConfig(**payload["model"]),
        train_populations=tuple(
            _population_from_dict(item) for item in payload["train_populations"]
        ),
        evaluation_populations=tuple(
            _population_from_dict(item) for item in payload["evaluation_populations"]
        ),
        arms=tuple(M1ArmConfig(**item) for item in payload["arms"]),
        seeds=tuple(payload["seeds"]),
        training=M1TrainingConfig(
            **{
                **payload["training"],
                "checkpoint_steps": tuple(payload["training"]["checkpoint_steps"]),
            }
        ),
        evaluation_examples=payload["evaluation_examples"],
        causal_examples=payload["causal_examples"],
        step_halving_seeds=tuple(payload.get("step_halving_seeds", ())),
    )


@dataclass(frozen=True)
class _RunSpec:
    arm_name: str
    qk_initial_scale: float
    seed: int
    learning_rate: float
    actual_steps: int
    actual_checkpoints: tuple[int, ...]
    time_divisor: int


def _run_specs(config: M1StudyConfig) -> tuple[_RunSpec, ...]:
    specs = [
        _RunSpec(
            arm_name=arm.name,
            qk_initial_scale=arm.qk_initial_scale,
            seed=seed,
            learning_rate=config.training.learning_rate,
            actual_steps=config.training.steps,
            actual_checkpoints=config.training.checkpoint_steps,
            time_divisor=1,
        )
        for arm in config.arms
        for seed in config.seeds
    ]
    standard = next((arm for arm in config.arms if arm.name == "standard"), None)
    if config.step_halving_seeds and standard is None:
        raise ValueError("step-halving audit requires a standard arm")
    if standard is not None:
        specs.extend(
            _RunSpec(
                arm_name="standard-step-half",
                qk_initial_scale=standard.qk_initial_scale,
                seed=seed,
                learning_rate=config.training.learning_rate / 2.0,
                actual_steps=2 * config.training.steps,
                actual_checkpoints=tuple(
                    2 * step for step in config.training.checkpoint_steps
                ),
                time_divisor=2,
            )
            for seed in config.step_halving_seeds
        )
    return tuple(specs)


def _fixed_batch(
    population: ZoologyMQARConfig,
    *,
    examples: int,
    seed: int,
    device: torch.device | str,
) -> MQARTokenBatch:
    return sample_zoology_mqar_batch(
        config=population,
        batch_size=examples,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device=device,
    )


def _factor_norms(model: M1Transformer) -> tuple[float, float]:
    qk_square = 0.0
    ov_square = 0.0
    for layer in model.layers:
        qk_square += float(layer.q_proj.weight.detach().square().sum().cpu())
        qk_square += float(layer.k_proj.weight.detach().square().sum().cpu())
        ov_square += float(layer.o_proj.weight.detach().square().sum().cpu())
        ov_square += float(layer.v_proj.weight.detach().square().sum().cpu())
    return sqrt(qk_square), sqrt(ov_square)


def _gradient_norms(model: M1Transformer, batch: MQARTokenBatch) -> tuple[float, float]:
    model.zero_grad(set_to_none=True)
    loss = model.query_loss(batch)
    loss.backward()
    qk_square = 0.0
    ov_square = 0.0
    for layer in model.layers:
        for parameter in (layer.q_proj.weight, layer.k_proj.weight):
            if parameter.grad is not None:
                qk_square += float(parameter.grad.detach().square().sum().cpu())
        for parameter in (layer.o_proj.weight, layer.v_proj.weight):
            if parameter.grad is not None:
                ov_square += float(parameter.grad.detach().square().sum().cpu())
    model.zero_grad(set_to_none=True)
    return sqrt(qk_square), sqrt(ov_square)


@torch.no_grad()
def _routing_metrics(
    model: M1Transformer,
    batch: MQARTokenBatch,
    *,
    arm_name: str,
    seed: int,
    step: int,
) -> list[dict[str, Any]]:
    hidden, trace = model(batch.input_ids, return_trace=True)
    rows = torch.arange(batch.batch_size, device=hidden.device)[:, None]
    query_hidden = hidden[rows, batch.query_positions]
    logits = torch.einsum("bmd,vd->bmv", query_hidden, model.output_weight)
    answers = batch.labels[rows, batch.query_positions]
    base_log_probability = F.log_softmax(logits, dim=-1).gather(-1, answers[..., None])[
        ..., 0
    ]

    deltas: list[torch.Tensor] = []
    for slot in range(batch.num_kv_pairs):
        mask = torch.zeros(
            batch.batch_size,
            batch.sequence_length,
            batch.sequence_length,
            dtype=torch.bool,
            device=hidden.device,
        )
        receiver = batch.query_positions
        key_source = batch.key_positions[:, slot][:, None].expand_as(receiver)
        value_source = batch.value_positions[:, slot][:, None].expand_as(receiver)
        mask[rows, receiver, key_source] = True
        mask[rows, receiver, value_source] = True
        blocked = model.query_logits(batch, edge_block_mask=mask)
        blocked_log_probability = F.log_softmax(blocked, dim=-1).gather(
            -1, answers[..., None]
        )[..., 0]
        deltas.append(base_log_probability - blocked_log_probability)
    slot_effect = torch.stack(deltas, dim=-1)
    diagonal = torch.eye(batch.num_kv_pairs, dtype=torch.bool, device=hidden.device)[
        None
    ]
    target_effect = slot_effect[diagonal.expand_as(slot_effect)]
    distractor_effect = slot_effect[~diagonal.expand_as(slot_effect)]
    s_key = target_effect.mean() - distractor_effect.mean()

    output: list[dict[str, Any]] = []
    pair_index = torch.arange(batch.num_kv_pairs, device=hidden.device)
    distractor_mask = ~torch.eye(
        batch.num_kv_pairs, dtype=torch.bool, device=hidden.device
    )
    for layer_index in range(model.config.num_layers):
        attention = trace[f"layers.{layer_index}.attention_probs"]
        scores = trace[f"layers.{layer_index}.qk_scores"]
        for head_index in range(model.config.num_heads):
            head_attention = attention[:, head_index]
            head_scores = scores[:, head_index]
            query_rows = head_attention[rows, batch.query_positions]
            query_scores = head_scores[rows, batch.query_positions]
            target_key = query_rows[rows, pair_index[None], batch.key_positions]
            target_value = query_rows[rows, pair_index[None], batch.value_positions]
            target_key_score = query_scores[rows, pair_index[None], batch.key_positions]
            all_key = query_rows[
                rows[:, :, None],
                pair_index[None, :, None],
                batch.key_positions[:, None, :],
            ]
            all_value = query_rows[
                rows[:, :, None],
                pair_index[None, :, None],
                batch.value_positions[:, None, :],
            ]
            all_key_score = query_scores[
                rows[:, :, None],
                pair_index[None, :, None],
                batch.key_positions[:, None, :],
            ]
            distractor_key = all_key[:, distractor_mask]
            distractor_value = all_value[:, distractor_mask]
            distractor_key_score = all_key_score[:, distractor_mask]
            output.append(
                {
                    "arm": arm_name,
                    "seed": seed,
                    "step": step,
                    "layer": layer_index,
                    "head": head_index,
                    "target_key_attention": float(target_key.mean().cpu()),
                    "distractor_key_attention": float(distractor_key.mean().cpu()),
                    "target_value_attention": float(target_value.mean().cpu()),
                    "distractor_value_attention": float(distractor_value.mean().cpu()),
                    "target_key_score_margin": float(
                        (target_key_score.mean() - distractor_key_score.mean()).cpu()
                    ),
                    "causal_slot_s_key": float(s_key.cpu()),
                    "causal_target_effect": float(target_effect.mean().cpu()),
                    "causal_distractor_effect": float(distractor_effect.mean().cpu()),
                    "path_scope": "query-row-to-full-card-direct-edges-all-layers",
                }
            )
    return output


def _metric_row(
    model: M1Transformer,
    batch: MQARTokenBatch,
    *,
    arm_name: str,
    seed: int,
    step: int,
    population: ZoologyMQARConfig,
    include_gradients: bool,
) -> dict[str, Any]:
    with torch.no_grad():
        logits = model.query_logits(batch)
        rows = torch.arange(batch.batch_size, device=batch.labels.device)[:, None]
        answers = batch.labels[rows, batch.query_positions]
        loss = F.cross_entropy(logits.flatten(0, 1), answers.flatten())
        accuracy = (logits.argmax(dim=-1) == answers).to(torch.float64).mean()
    qk_norm, ov_norm = _factor_norms(model)
    if include_gradients:
        qk_gradient, ov_gradient = _gradient_norms(model, batch)
    else:
        qk_gradient = None
        ov_gradient = None
    return {
        "arm": arm_name,
        "seed": seed,
        "step": step,
        "sequence_length": population.sequence_length,
        "num_kv_pairs": population.num_kv_pairs,
        "nll": float(loss.cpu()),
        "accuracy": float(accuracy.cpu()),
        "qk_factor_norm": qk_norm,
        "ov_factor_norm": ov_norm,
        "qk_gradient_norm": qk_gradient,
        "ov_gradient_norm": ov_gradient,
    }


def _train_one(
    config: M1StudyConfig,
    spec: _RunSpec,
    *,
    device: torch.device | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_config = replace(config.model, qk_initial_scale=spec.qk_initial_scale)
    cuda_devices: list[int] = []
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        cuda_devices = [resolved_device.index or 0]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(_seed_key(config.study_id, "init", spec.seed))
        if resolved_device.type == "cuda":
            torch.cuda.manual_seed_all(_seed_key(config.study_id, "init", spec.seed))
        model = M1Transformer(model_config).to(resolved_device)
    if config.training.optimizer.lower() == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(), lr=spec.learning_rate
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=spec.learning_rate, weight_decay=0.0
        )
    checkpoints = set(spec.actual_checkpoints)
    metrics: list[dict[str, Any]] = []
    routing: list[dict[str, Any]] = []
    model.train()

    for actual_step in range(spec.actual_steps + 1):
        nominal_step = actual_step // spec.time_divisor
        if actual_step in checkpoints:
            model.eval()
            populations = (
                config.evaluation_populations
                if actual_step == spec.actual_steps
                else config.evaluation_populations[:1]
            )
            for population_index, population in enumerate(populations):
                evaluation_batch = _fixed_batch(
                    population,
                    examples=config.evaluation_examples,
                    seed=_seed_key(
                        config.study_id, "eval", spec.seed, population_index
                    ),
                    device=resolved_device,
                )
                metrics.append(
                    _metric_row(
                        model,
                        evaluation_batch,
                        arm_name=spec.arm_name,
                        seed=spec.seed,
                        step=nominal_step,
                        population=population,
                        include_gradients=population_index == 0,
                    )
                )
            causal_population = config.evaluation_populations[0]
            causal_batch = _fixed_batch(
                causal_population,
                examples=config.causal_examples,
                seed=_seed_key(config.study_id, "causal", spec.seed),
                device=resolved_device,
            )
            routing.extend(
                _routing_metrics(
                    model,
                    causal_batch,
                    arm_name=spec.arm_name,
                    seed=spec.seed,
                    step=nominal_step,
                )
            )
            model.train()
        if actual_step == spec.actual_steps:
            break

        population_index = (nominal_step + spec.seed) % len(config.train_populations)
        population = config.train_populations[population_index]
        batch_size = max(1, config.training.batch_tokens // population.sequence_length)
        training_batch = sample_zoology_mqar_batch(
            config=population,
            batch_size=batch_size,
            generator=torch.Generator(device="cpu").manual_seed(
                _seed_key(config.study_id, "train", spec.seed, nominal_step)
            ),
            device=resolved_device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.query_loss(training_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"nonfinite training loss in {spec.arm_name} seed {spec.seed}"
            )
        loss.backward()
        optimizer.step()
    return metrics, routing


def _run_directory(root: Path, spec: _RunSpec) -> Path:
    return root / "runs" / spec.arm_name / f"seed-{spec.seed}"


def _write_run(
    directory: Path,
    *,
    config_hash: str,
    sources: dict[str, str],
    spec: _RunSpec,
    metrics: list[dict[str, Any]],
    routing: list[dict[str, Any]],
) -> None:
    metrics_bytes = _canonical_bytes(metrics)
    routing_bytes = _canonical_bytes(routing)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_config_hash": config_hash,
        "source_hashes": sources,
        "run_spec": asdict(spec),
        "receipts": {
            "metrics.json": _hash_bytes(metrics_bytes),
            "routing.json": _hash_bytes(routing_bytes),
        },
    }
    _atomic_write(directory / "metrics.json", metrics_bytes)
    _atomic_write(directory / "routing.json", routing_bytes)
    _atomic_write(directory / "manifest.json", _canonical_bytes(manifest))
    _atomic_write(directory / "_SUCCESS", _canonical_bytes({"complete": True}))


def _validate_run(
    directory: Path,
    *,
    config_hash: str,
    sources: dict[str, str],
    spec: _RunSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = ("metrics.json", "routing.json", "manifest.json", "_SUCCESS")
    if not all((directory / name).is_file() for name in required):
        raise ValueError(f"incomplete M1 run: {directory}")
    manifest = json.loads((directory / "manifest.json").read_text())
    expected_run_spec = json.loads(_canonical_bytes(asdict(spec)))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("study_config_hash") != config_hash
        or manifest.get("source_hashes") != sources
        or manifest.get("run_spec") != expected_run_spec
    ):
        raise ValueError(f"M1 run identity mismatch: {directory}")
    for name, digest in manifest["receipts"].items():
        if _hash_file(directory / name) != digest:
            raise ValueError(f"M1 run receipt mismatch for {directory / name}")
    metrics = json.loads((directory / "metrics.json").read_text())
    routing = json.loads((directory / "routing.json").read_text())
    for row in metrics:
        if not 0.0 <= row["accuracy"] <= 1.0:
            raise ValueError("accuracy lies outside [0,1]")
        if row["nll"] < 0.0 or not isfinite(row["nll"]):
            raise ValueError("invalid NLL in M1 metrics")
    return metrics, routing


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _materialize_root(
    config: M1StudyConfig,
    root: Path,
    *,
    config_hash: str,
    sources: dict[str, str],
) -> M1ValidatedArtifact:
    all_metrics: list[dict[str, Any]] = []
    all_routing: list[dict[str, Any]] = []
    specs = _run_specs(config)
    run_receipts: dict[str, str] = {}
    for spec in specs:
        directory = _run_directory(root, spec)
        metrics, routing = _validate_run(
            directory,
            config_hash=config_hash,
            sources=sources,
            spec=spec,
        )
        all_metrics.extend(metrics)
        all_routing.extend(routing)
        relative = directory.relative_to(root).as_posix()
        run_receipts[relative] = _hash_file(directory / "manifest.json")
    metrics_bytes = _canonical_bytes(all_metrics)
    routing_bytes = _canonical_bytes(all_routing)
    files = {
        "metrics.json": metrics_bytes,
        "metrics.csv": _csv_bytes(all_metrics),
        "routing.json": routing_bytes,
        "routing.csv": _csv_bytes(all_routing),
    }
    for name, content in files.items():
        _atomic_write(root / name, content)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config": asdict(config),
        "study_config_hash": config_hash,
        "source_hashes": sources,
        "independent_unit": "training_seed",
        "repeated_measures": [
            "arm",
            "checkpoint",
            "population",
            "layer",
            "head",
            "query",
        ],
        "run_manifest_receipts": run_receipts,
        "root_receipts": {
            name: _hash_bytes(content) for name, content in files.items()
        },
    }
    _atomic_write(root / "manifest.json", _canonical_bytes(manifest))
    _atomic_write(root / "_SUCCESS", _canonical_bytes({"complete": True}))
    return M1ValidatedArtifact(
        study_id=config.study_id,
        seed_runs=len(specs),
        metric_rows=len(all_metrics),
        routing_rows=len(all_routing),
    )


def validate_m1_artifact(root: Path) -> M1ValidatedArtifact:
    """Reconstruct every aggregate from seed artifacts and verify all hashes."""

    if not (root / "_SUCCESS").is_file():
        raise ValueError("M1 artifact has no root _SUCCESS")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported M1 artifact schema")
    config_payload = manifest["study_config"]
    config = M1StudyConfig(
        study_id=config_payload["study_id"],
        upstream_commit=config_payload["upstream_commit"],
        upstream_source_sha256=config_payload["upstream_source_sha256"],
        model=M1ModelConfig(**config_payload["model"]),
        train_populations=tuple(
            ZoologyMQARConfig(**item) for item in config_payload["train_populations"]
        ),
        evaluation_populations=tuple(
            ZoologyMQARConfig(**item)
            for item in config_payload["evaluation_populations"]
        ),
        arms=tuple(M1ArmConfig(**item) for item in config_payload["arms"]),
        seeds=tuple(config_payload["seeds"]),
        training=M1TrainingConfig(
            **{
                **config_payload["training"],
                "checkpoint_steps": tuple(
                    config_payload["training"]["checkpoint_steps"]
                ),
            }
        ),
        evaluation_examples=config_payload["evaluation_examples"],
        causal_examples=config_payload["causal_examples"],
        step_halving_seeds=tuple(config_payload["step_halving_seeds"]),
    )
    config_hash = _study_hash(config)
    sources = _source_hashes()
    if config_hash != manifest.get("study_config_hash") or sources != manifest.get(
        "source_hashes"
    ):
        raise ValueError("M1 root identity or source hash mismatch")

    expected_metrics: list[dict[str, Any]] = []
    expected_routing: list[dict[str, Any]] = []
    for spec in _run_specs(config):
        metrics, routing = _validate_run(
            _run_directory(root, spec),
            config_hash=config_hash,
            sources=sources,
            spec=spec,
        )
        expected_metrics.extend(metrics)
        expected_routing.extend(routing)
    expected_files = {
        "metrics.json": _canonical_bytes(expected_metrics),
        "metrics.csv": _csv_bytes(expected_metrics),
        "routing.json": _canonical_bytes(expected_routing),
        "routing.csv": _csv_bytes(expected_routing),
    }
    for name, content in expected_files.items():
        if (root / name).read_bytes() != content:
            raise ValueError(f"M1 aggregate does not reconstruct from raw runs: {name}")
        if manifest["root_receipts"].get(name) != _hash_bytes(content):
            raise ValueError(f"M1 root receipt mismatch: {name}")
    return M1ValidatedArtifact(
        study_id=config.study_id,
        seed_runs=len(_run_specs(config)),
        metric_rows=len(expected_metrics),
        routing_rows=len(expected_routing),
    )


def run_m1_study(
    config: M1StudyConfig,
    *,
    output_directory: Path,
    device: torch.device | str,
) -> M1RunSummary:
    """Run missing seed arms, content-check completed ones, and rebuild the root."""

    output_directory.mkdir(parents=True, exist_ok=True)
    config_hash = _study_hash(config)
    sources = _source_hashes()
    completed = 0
    skipped = 0
    specs = _run_specs(config)
    for spec in specs:
        directory = _run_directory(output_directory, spec)
        if (directory / "_SUCCESS").is_file():
            _validate_run(
                directory,
                config_hash=config_hash,
                sources=sources,
                spec=spec,
            )
            skipped += 1
            continue
        metrics, routing = _train_one(config, spec, device=device)
        _write_run(
            directory,
            config_hash=config_hash,
            sources=sources,
            spec=spec,
            metrics=metrics,
            routing=routing,
        )
        completed += 1
        print(f"[mqar-m1] completed arm={spec.arm_name} seed={spec.seed}", flush=True)
    _materialize_root(
        config,
        output_directory,
        config_hash=config_hash,
        sources=sources,
    )
    return M1RunSummary(
        study_id=config.study_id,
        planned_runs=len(specs),
        completed_runs=completed,
        skipped_runs=skipped,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    config = load_m1_study_config(arguments.config)
    summary = run_m1_study(
        config,
        output_directory=arguments.output_directory,
        device=arguments.device,
    )
    print(json.dumps(asdict(summary), sort_keys=True))


if __name__ == "__main__":
    _main()
