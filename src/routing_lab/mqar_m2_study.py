"""Resumable seed-grain runner for the frozen MQAR M2 experiment.

Each training seed is independent.  Arms share the same raw initialization and
counter-addressed batches; only the registered Q/K relation and scale differ.
Artifacts are content-addressed at run and root level so a partial or mixed-contract
study cannot be analyzed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import StringIO
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
from .mqar_m2 import M2ArmConfig, initialize_m2_model, measure_qk_geometry

SCHEMA_VERSION = "mqar-m2-orientation-study-v1"


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


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _seed_key(study_id: str, *parts: object) -> int:
    message = ":".join(map(str, (SCHEMA_VERSION, study_id, *parts))).encode("utf-8")
    return int.from_bytes(sha256(message).digest()[:8], "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class M2TrainingConfig:
    """Frozen finite-step optimizer and checkpoint contract."""

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
class M2StudyConfig:
    """Complete scientific identity of the M2 signed-orientation study."""

    study_id: str
    upstream_commit: str
    upstream_source_sha256: str
    model: M1ModelConfig
    train_populations: tuple[ZoologyMQARConfig, ...]
    evaluation_populations: tuple[ZoologyMQARConfig, ...]
    arms: tuple[M2ArmConfig, ...]
    seeds: tuple[int, ...]
    training: M2TrainingConfig
    evaluation_examples: int
    routing_examples: int

    def __post_init__(self) -> None:
        if not self.study_id.strip():
            raise ValueError("study_id must be nonempty")
        if len(self.upstream_commit) != 40 or len(self.upstream_source_sha256) < 12:
            raise ValueError("upstream source identity is incomplete")
        if not self.train_populations or not self.evaluation_populations:
            raise ValueError("training and evaluation populations are required")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("training seeds must be nonempty and unique")
        names = tuple(arm.name for arm in self.arms)
        if len(names) != 5 or len(set(names)) != 5:
            raise ValueError("M2 requires exactly five uniquely named arms")
        expected = {
            "independent": ("independent", 1.0),
            "positive": ("tied-positive", 1.0),
            "negative": ("tied-negative", 1.0),
            "positive-small": ("tied-positive", 2.0**-8),
            "negative-small": ("tied-negative", 2.0**-8),
        }
        observed = {arm.name: (arm.relation, arm.qk_initial_scale) for arm in self.arms}
        if observed != expected:
            raise ValueError(
                "M2 arms are not correctly named for the frozen sign-by-scale design"
            )
        if self.evaluation_examples < 1 or self.routing_examples < 1:
            raise ValueError("evaluation and routing example counts must be positive")
        all_populations = (*self.train_populations, *self.evaluation_populations)
        if any(item.vocab_size != self.model.vocab_size for item in all_populations):
            raise ValueError("population and model vocabularies must agree")
        if any(
            item.sequence_length > self.model.max_sequence_length
            for item in all_populations
        ):
            raise ValueError("population sequence length exceeds the model limit")


@dataclass(frozen=True)
class M2RunSummary:
    study_id: str
    planned_runs: int
    completed_runs: int
    skipped_runs: int


@dataclass(frozen=True)
class M2ValidatedArtifact:
    study_id: str
    seed_runs: int
    metric_rows: int
    geometry_rows: int


@dataclass(frozen=True)
class _RunSpec:
    arm_name: str
    relation: str
    qk_initial_scale: float
    seed: int


def _study_hash(config: M2StudyConfig) -> str:
    return _hash_bytes(_canonical_bytes(asdict(config)))


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: _hash_file(directory / name)
        for name in ("mqar_m1.py", "mqar_m2.py", "mqar_m2_study.py")
    }


def _population(payload: dict[str, Any]) -> ZoologyMQARConfig:
    return ZoologyMQARConfig(**payload)


def _config_from_payload(payload: dict[str, Any]) -> M2StudyConfig:
    return M2StudyConfig(
        study_id=payload["study_id"],
        upstream_commit=payload["upstream_commit"],
        upstream_source_sha256=payload["upstream_source_sha256"],
        model=M1ModelConfig(**payload["model"]),
        train_populations=tuple(
            _population(item) for item in payload["train_populations"]
        ),
        evaluation_populations=tuple(
            _population(item) for item in payload["evaluation_populations"]
        ),
        arms=tuple(M2ArmConfig(**item) for item in payload["arms"]),
        seeds=tuple(payload["seeds"]),
        training=M2TrainingConfig(
            **{
                **payload["training"],
                "checkpoint_steps": tuple(payload["training"]["checkpoint_steps"]),
            }
        ),
        evaluation_examples=payload["evaluation_examples"],
        routing_examples=payload["routing_examples"],
    )


def load_m2_study_config(path: Path) -> M2StudyConfig:
    """Load the immutable JSON design into validated dataclasses."""

    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("M2 config schema mismatch")
    payload = {key: value for key, value in payload.items() if key != "schema_version"}
    return _config_from_payload(payload)


def _run_specs(config: M2StudyConfig) -> tuple[_RunSpec, ...]:
    return tuple(
        _RunSpec(
            arm_name=arm.name,
            relation=arm.relation,
            qk_initial_scale=arm.qk_initial_scale,
            seed=seed,
        )
        for arm in config.arms
        for seed in config.seeds
    )


def _execution_environment(device: torch.device | str) -> dict[str, Any]:
    resolved = torch.device(device)
    gpu_name: str | None = None
    capability: list[int] | None = None
    if resolved.type == "cuda":
        index = resolved.index or 0
        gpu_name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_type": resolved.type,
        "gpu_name": gpu_name,
        "gpu_capability": capability,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None
        ),
    }


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


def _qk_norm(model: M1Transformer) -> float:
    total = sum(
        float(parameter.detach().square().sum().cpu())
        for layer in model.layers
        for parameter in (layer.q_proj.weight, layer.k_proj.weight)
    )
    return sqrt(total)


def _qk_gradient_norm(model: M1Transformer, batch: MQARTokenBatch) -> float:
    model.zero_grad(set_to_none=True)
    model.query_loss(batch).backward()
    total = sum(
        float(parameter.grad.detach().square().sum().cpu())
        for layer in model.layers
        for parameter in (layer.q_proj.weight, layer.k_proj.weight)
        if parameter.grad is not None
    )
    model.zero_grad(set_to_none=True)
    return sqrt(total)


def _metric_row(
    model: M1Transformer,
    batch: MQARTokenBatch,
    *,
    spec: _RunSpec,
    step: int,
    population: ZoologyMQARConfig,
    include_gradient: bool,
) -> dict[str, Any]:
    with torch.no_grad():
        logits = model.query_logits(batch)
        rows = torch.arange(batch.batch_size, device=batch.labels.device)[:, None]
        answers = batch.labels[rows, batch.query_positions]
        nll = F.cross_entropy(logits.flatten(0, 1), answers.flatten())
        accuracy = (logits.argmax(dim=-1) == answers).to(torch.float64).mean()
    return {
        "arm": spec.arm_name,
        "relation": spec.relation,
        "qk_initial_scale": spec.qk_initial_scale,
        "seed": spec.seed,
        "step": step,
        "sequence_length": population.sequence_length,
        "num_kv_pairs": population.num_kv_pairs,
        "nll": float(nll.cpu()),
        "accuracy": float(accuracy.cpu()),
        "qk_factor_norm": _qk_norm(model),
        "qk_gradient_norm": (
            _qk_gradient_norm(model, batch) if include_gradient else None
        ),
    }


@torch.no_grad()
def _geometry_rows(
    model: M1Transformer,
    batch: MQARTokenBatch,
    *,
    spec: _RunSpec,
    step: int,
) -> list[dict[str, Any]]:
    _hidden, trace = model(batch.input_ids, return_trace=True)
    rows = torch.arange(batch.batch_size, device=batch.input_ids.device)[:, None]
    target_index = torch.arange(batch.num_kv_pairs, device=batch.input_ids.device)
    distractor_mask = ~torch.eye(
        batch.num_kv_pairs, dtype=torch.bool, device=batch.input_ids.device
    )
    factor_geometry = {(row.layer, row.head): row for row in measure_qk_geometry(model)}
    output: list[dict[str, Any]] = []
    for layer_index in range(model.config.num_layers):
        attention = trace[f"layers.{layer_index}.attention_probs"]
        scores = trace[f"layers.{layer_index}.qk_scores"]
        for head_index in range(model.config.num_heads):
            query_attention = attention[:, head_index][rows, batch.query_positions]
            query_scores = scores[:, head_index][rows, batch.query_positions]
            target_attention = query_attention[
                rows, target_index[None], batch.key_positions
            ]
            target_score = query_scores[rows, target_index[None], batch.key_positions]
            all_attention = query_attention[
                rows[:, :, None],
                target_index[None, :, None],
                batch.key_positions[:, None, :],
            ]
            all_scores = query_scores[
                rows[:, :, None],
                target_index[None, :, None],
                batch.key_positions[:, None, :],
            ]
            distractor_attention = all_attention[:, distractor_mask]
            distractor_score = all_scores[:, distractor_mask]
            geometry = factor_geometry[(layer_index, head_index)]
            output.append(
                {
                    "arm": spec.arm_name,
                    "relation": spec.relation,
                    "qk_initial_scale": spec.qk_initial_scale,
                    "seed": spec.seed,
                    "step": step,
                    "layer": layer_index,
                    "head": head_index,
                    "qk_factor_cosine": geometry.qk_factor_cosine,
                    "normalized_composite_trace": (geometry.normalized_composite_trace),
                    "composite_skew_fraction": geometry.composite_skew_fraction,
                    "target_key_attention": float(target_attention.mean().cpu()),
                    "distractor_key_attention": float(
                        distractor_attention.mean().cpu()
                    ),
                    "target_key_score_margin": float(
                        (target_score.mean() - distractor_score.mean()).cpu()
                    ),
                }
            )
    return output


def _train_one(
    config: M2StudyConfig,
    spec: _RunSpec,
    *,
    device: torch.device | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    arm = next(arm for arm in config.arms if arm.name == spec.arm_name)
    initialized = initialize_m2_model(
        config.model,
        arm=arm,
        initialization_seed=_seed_key(config.study_id, "init", spec.seed),
        device=device,
    )
    model = initialized.model
    if config.training.optimizer.lower() == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(), lr=config.training.learning_rate
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.training.learning_rate, weight_decay=0.0
        )
    checkpoints = set(config.training.checkpoint_steps)
    metrics: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    model.train()

    for step in range(config.training.steps + 1):
        if step in checkpoints:
            model.eval()
            populations = (
                config.evaluation_populations
                if step == config.training.steps
                else config.evaluation_populations[:2]
            )
            for population_index, population in enumerate(populations):
                evaluation_batch = _fixed_batch(
                    population,
                    examples=config.evaluation_examples,
                    seed=_seed_key(
                        config.study_id, "eval", spec.seed, population_index
                    ),
                    device=device,
                )
                metrics.append(
                    _metric_row(
                        model,
                        evaluation_batch,
                        spec=spec,
                        step=step,
                        population=population,
                        include_gradient=population_index == 0,
                    )
                )
            routing_batch = _fixed_batch(
                config.evaluation_populations[0],
                examples=config.routing_examples,
                seed=_seed_key(config.study_id, "routing", spec.seed),
                device=device,
            )
            geometry.extend(_geometry_rows(model, routing_batch, spec=spec, step=step))
            model.train()
        if step == config.training.steps:
            break

        population_index = (step + spec.seed) % len(config.train_populations)
        population = config.train_populations[population_index]
        batch_size = max(1, config.training.batch_tokens // population.sequence_length)
        training_batch = sample_zoology_mqar_batch(
            config=population,
            batch_size=batch_size,
            generator=torch.Generator(device="cpu").manual_seed(
                _seed_key(config.study_id, "train", spec.seed, step)
            ),
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.query_loss(training_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"nonfinite training loss in {spec.arm_name} seed {spec.seed}"
            )
        loss.backward()
        optimizer.step()

    return metrics, geometry, asdict(initialized.audit)


def _run_directory(root: Path, spec: _RunSpec) -> Path:
    return root / "runs" / spec.arm_name / f"seed-{spec.seed}"


def _write_run(
    directory: Path,
    *,
    config_hash: str,
    sources: dict[str, str],
    environment_hash: str,
    spec: _RunSpec,
    metrics: list[dict[str, Any]],
    geometry: list[dict[str, Any]],
    initialization_audit: dict[str, Any],
) -> None:
    metric_bytes = _canonical_bytes(metrics)
    geometry_bytes = _canonical_bytes(geometry)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_config_hash": config_hash,
        "source_hashes": sources,
        "execution_environment_sha256": environment_hash,
        "run_spec": asdict(spec),
        "initialization_audit": initialization_audit,
        "receipts": {
            "metrics.json": _hash_bytes(metric_bytes),
            "geometry.json": _hash_bytes(geometry_bytes),
        },
    }
    _atomic_write(directory / "metrics.json", metric_bytes)
    _atomic_write(directory / "geometry.json", geometry_bytes)
    _atomic_write(directory / "manifest.json", _canonical_bytes(manifest))
    _atomic_write(directory / "_SUCCESS", b"m2-run-complete\n")


def _validate_run(
    directory: Path,
    *,
    config: M2StudyConfig,
    config_hash: str,
    sources: dict[str, str],
    environment_hash: str,
    spec: _RunSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    required = ("metrics.json", "geometry.json", "manifest.json", "_SUCCESS")
    if not all((directory / name).is_file() for name in required):
        raise ValueError(f"incomplete M2 run: {directory}")
    if (directory / "_SUCCESS").read_bytes() != b"m2-run-complete\n":
        raise ValueError(f"malformed M2 run success marker: {directory}")
    manifest = json.loads((directory / "manifest.json").read_text())
    expected_spec = json.loads(_canonical_bytes(asdict(spec)))
    identity = (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("study_config_hash") == config_hash
        and manifest.get("source_hashes") == sources
        and manifest.get("execution_environment_sha256") == environment_hash
        and manifest.get("run_spec") == expected_spec
    )
    if not identity:
        raise ValueError(f"M2 run identity mismatch: {directory}")
    for name, expected in manifest["receipts"].items():
        if _hash_file(directory / name) != expected:
            raise ValueError(f"M2 run receipt mismatch: {directory / name}")

    metrics = json.loads((directory / "metrics.json").read_text())
    geometry = json.loads((directory / "geometry.json").read_text())
    metric_keys = {
        (
            row["step"],
            row["sequence_length"],
            row["num_kv_pairs"],
        )
        for row in metrics
    }
    expected_metric_keys: set[tuple[int, int, int]] = set()
    for step in config.training.checkpoint_steps:
        populations = (
            config.evaluation_populations
            if step == config.training.steps
            else config.evaluation_populations[:2]
        )
        expected_metric_keys.update(
            (step, item.sequence_length, item.num_kv_pairs) for item in populations
        )
    if len(metric_keys) != len(metrics) or metric_keys != expected_metric_keys:
        raise ValueError(f"M2 metric grid mismatch: {directory}")
    for row in metrics:
        if not 0.0 <= row["accuracy"] <= 1.0:
            raise ValueError("M2 accuracy lies outside [0,1]")
        if row["nll"] < 0.0 or not isfinite(row["nll"]):
            raise ValueError("invalid M2 NLL")

    geometry_keys = {(row["step"], row["layer"], row["head"]) for row in geometry}
    expected_geometry_keys = {
        (step, layer, head)
        for step in config.training.checkpoint_steps
        for layer in range(config.model.num_layers)
        for head in range(config.model.num_heads)
    }
    if len(geometry_keys) != len(geometry) or geometry_keys != expected_geometry_keys:
        raise ValueError(f"M2 geometry grid mismatch: {directory}")
    for row in geometry:
        if not all(
            isfinite(row[name])
            for name in (
                "qk_factor_cosine",
                "normalized_composite_trace",
                "composite_skew_fraction",
                "target_key_attention",
                "distractor_key_attention",
                "target_key_score_margin",
            )
        ):
            raise ValueError("nonfinite M2 geometry value")

    initialization_audit = manifest["initialization_audit"]
    if initialization_audit["max_relation_error"] > 1.0e-12:
        raise ValueError("M2 initialization relation audit failed")
    if initialization_audit["max_scale_error"] > 1.0e-12:
        raise ValueError("M2 initialization scale audit failed")
    if spec.relation != "independent":
        sign = 1.0 if spec.relation == "tied-positive" else -1.0
        step_zero = [row for row in geometry if row["step"] == 0]
        if any(abs(row["qk_factor_cosine"] - sign) > 1.0e-12 for row in step_zero):
            raise ValueError("M2 step-zero signed geometry mismatch")
        if any(sign * row["normalized_composite_trace"] <= 0.0 for row in step_zero):
            raise ValueError("M2 step-zero composite trace has the wrong sign")
        if any(row["composite_skew_fraction"] > 1.0e-12 for row in step_zero):
            raise ValueError("M2 step-zero tied composite is not symmetric")
    return metrics, geometry, initialization_audit


def _pairing_audit(
    config: M2StudyConfig,
    audits: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for seed in config.seeds:
        rows = [audits[(arm.name, seed)] for arm in config.arms]
        base_q = {row["base_q_sha256"] for row in rows}
        base_k = {row["base_k_sha256"] for row in rows}
        non_qk = {row["non_qk_sha256"] for row in rows}
        by_name = {row["arm_name"]: row for row in rows}
        q_pairing = (
            by_name["positive"]["initialized_q_sha256"]
            == by_name["negative"]["initialized_q_sha256"]
            and by_name["positive-small"]["initialized_q_sha256"]
            == by_name["negative-small"]["initialized_q_sha256"]
        )
        max_error = max(row["max_relation_error"] for row in rows)
        max_scale_error = max(row["max_scale_error"] for row in rows)
        pairing_pass = (
            len(base_q) == len(base_k) == len(non_qk) == 1
            and q_pairing
            and max_error <= 1.0e-12
            and max_scale_error <= 1.0e-12
        )
        if not pairing_pass:
            raise ValueError(f"M2 initialization pairing failed for seed {seed}")
        output[str(seed)] = {
            "pairing_pass": True,
            "base_q_sha256": next(iter(base_q)),
            "base_k_sha256": next(iter(base_k)),
            "non_qk_sha256": next(iter(non_qk)),
            "max_relation_error": max_error,
            "max_scale_error": max_scale_error,
        }
    return output


def _materialize_root(
    config: M2StudyConfig,
    root: Path,
    *,
    config_hash: str,
    sources: dict[str, str],
    environment_hash: str,
) -> M2ValidatedArtifact:
    all_metrics: list[dict[str, Any]] = []
    all_geometry: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], dict[str, Any]] = {}
    run_receipts: dict[str, str] = {}
    for spec in _run_specs(config):
        directory = _run_directory(root, spec)
        metrics, geometry, audit = _validate_run(
            directory,
            config=config,
            config_hash=config_hash,
            sources=sources,
            environment_hash=environment_hash,
            spec=spec,
        )
        all_metrics.extend(metrics)
        all_geometry.extend(geometry)
        audits[(spec.arm_name, spec.seed)] = audit
        run_receipts[directory.relative_to(root).as_posix()] = _hash_file(
            directory / "manifest.json"
        )

    files = {
        "metrics.json": _canonical_bytes(all_metrics),
        "metrics.csv": _csv_bytes(all_metrics),
        "geometry.json": _canonical_bytes(all_geometry),
        "geometry.csv": _csv_bytes(all_geometry),
    }
    for name, content in files.items():
        _atomic_write(root / name, content)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config": asdict(config),
        "study_config_hash": config_hash,
        "source_hashes": sources,
        "execution_environment_sha256": environment_hash,
        "independent_unit": "training_seed",
        "repeated_measures": [
            "arm",
            "checkpoint",
            "population",
            "layer",
            "head",
            "example",
        ],
        "initialization_pairing_audit": _pairing_audit(config, audits),
        "run_manifest_receipts": run_receipts,
        "root_receipts": {
            name: _hash_bytes(content) for name, content in files.items()
        },
    }
    _atomic_write(root / "manifest.json", _canonical_bytes(manifest))
    _atomic_write(root / "_SUCCESS", b"m2-study-complete\n")
    return M2ValidatedArtifact(
        study_id=config.study_id,
        seed_runs=len(_run_specs(config)),
        metric_rows=len(all_metrics),
        geometry_rows=len(all_geometry),
    )


def validate_m2_artifact(root: Path) -> M2ValidatedArtifact:
    """Rebuild every root file from run artifacts and verify source identity."""

    if (root / "_SUCCESS").read_bytes() != b"m2-study-complete\n":
        raise ValueError("M2 root success marker is absent or malformed")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported M2 artifact schema")
    config = _config_from_payload(manifest["study_config"])
    config_hash = _study_hash(config)
    sources = _source_hashes()
    environment_hash = _hash_file(root / "execution_environment.json")
    if (
        manifest.get("study_config_hash") != config_hash
        or manifest.get("source_hashes") != sources
        or manifest.get("execution_environment_sha256") != environment_hash
    ):
        raise ValueError("M2 root identity or source hash mismatch")

    all_metrics: list[dict[str, Any]] = []
    all_geometry: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in _run_specs(config):
        metrics, geometry, audit = _validate_run(
            _run_directory(root, spec),
            config=config,
            config_hash=config_hash,
            sources=sources,
            environment_hash=environment_hash,
            spec=spec,
        )
        all_metrics.extend(metrics)
        all_geometry.extend(geometry)
        audits[(spec.arm_name, spec.seed)] = audit
    pairing = _pairing_audit(config, audits)
    if manifest.get("initialization_pairing_audit") != pairing:
        raise ValueError("M2 root initialization pairing audit mismatch")

    expected_files = {
        "metrics.json": _canonical_bytes(all_metrics),
        "metrics.csv": _csv_bytes(all_metrics),
        "geometry.json": _canonical_bytes(all_geometry),
        "geometry.csv": _csv_bytes(all_geometry),
    }
    for name, content in expected_files.items():
        if (root / name).read_bytes() != content:
            raise ValueError(f"M2 aggregate does not reconstruct from runs: {name}")
        if manifest["root_receipts"].get(name) != _hash_bytes(content):
            raise ValueError(f"M2 root receipt mismatch: {name}")
    return M2ValidatedArtifact(
        study_id=config.study_id,
        seed_runs=len(_run_specs(config)),
        metric_rows=len(all_metrics),
        geometry_rows=len(all_geometry),
    )


def run_m2_study(
    config: M2StudyConfig,
    *,
    output_directory: Path,
    device: torch.device | str,
) -> M2RunSummary:
    """Run missing paired arms, validate completed arms, and rebuild the root."""

    output_directory.mkdir(parents=True, exist_ok=True)
    environment_bytes = _canonical_bytes(_execution_environment(device))
    environment_path = output_directory / "execution_environment.json"
    if (
        environment_path.is_file()
        and environment_path.read_bytes() != environment_bytes
    ):
        raise ValueError("M2 output directory belongs to another execution environment")
    _atomic_write(environment_path, environment_bytes)
    environment_hash = _hash_bytes(environment_bytes)
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
                config=config,
                config_hash=config_hash,
                sources=sources,
                environment_hash=environment_hash,
                spec=spec,
            )
            skipped += 1
            continue
        metrics, geometry, audit = _train_one(config, spec, device=device)
        _write_run(
            directory,
            config_hash=config_hash,
            sources=sources,
            environment_hash=environment_hash,
            spec=spec,
            metrics=metrics,
            geometry=geometry,
            initialization_audit=audit,
        )
        completed += 1
        print(f"[mqar-m2] completed arm={spec.arm_name} seed={spec.seed}", flush=True)
    _materialize_root(
        config,
        output_directory,
        config_hash=config_hash,
        sources=sources,
        environment_hash=environment_hash,
    )
    return M2RunSummary(
        study_id=config.study_id,
        planned_runs=len(specs),
        completed_runs=completed,
        skipped_runs=skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    summary = run_m2_study(
        load_m2_study_config(arguments.config),
        output_directory=arguments.output_directory,
        device=arguments.device,
    )
    print(json.dumps(asdict(summary), sort_keys=True))


if __name__ == "__main__":
    main()
