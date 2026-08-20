"""Fail-closed finer-Euler remedy for population-GF trajectories that fail P38.

The original P36 run is never overwritten.  Given a completed failed source and
refinement factor ``q``, this module integrates the nested triplet

``eta0/q, eta0/(2q), eta0/(4q)``

over exactly the same physical horizon and at exactly the same observation times.
The public rows keep normalized divisors ``1,2,4`` so the existing P38 machinery
can be reused; the manifest records their actual divisors relative to the original
P36 step.  A refined directory is a numerical remedy, not evidence that the
original registered discretization passed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from .control_config import CodebookConfig, CompositeConfig, canonical_sha256
from .controlled_model import ControlledModelConfig
from .population_gf import enumerate_retrieval_population
from .population_gf_study import (
    GF_ORDER_PARAMETER_NAMES,
    SCHEMA_VERSION,
    PopulationGFStudyConfig,
    _initialize_model,
    _model_state_hash,
    _run_or_resume_trajectory,
    _touch_success,
    _write_csv,
    _write_json,
    compute_step_halving_audit,
)

_REQUIRED_SOURCE_FILES = (
    "_SUCCESS",
    "manifest.json",
    "study_config.json",
    "initial_hessian.json",
    "trajectory.json",
    "trajectory.csv",
    "step_halving.json",
)


@dataclass(frozen=True)
class PopulationGFRefinementConfig:
    """Identity of one nested finer-step remedy."""

    study_id: str
    source_directory: str
    refinement_factor: int

    def __post_init__(self) -> None:
        if not self.study_id.strip() or not self.source_directory:
            raise ValueError("study_id and source_directory must be nonempty")
        if self.refinement_factor < 2 or self.refinement_factor & (
            self.refinement_factor - 1
        ):
            raise ValueError("refinement_factor must be a power of two at least two")


@dataclass(frozen=True)
class PopulationGFRefinementResult:
    output_directory: Path
    refinement_config_hash: str
    completed_trajectories: int
    skipped_trajectories: int
    trajectory_rows: int
    gf_like_discretization_pass: bool


def _model_config_from_mapping(payload: dict[str, Any]) -> ControlledModelConfig:
    values = dict(payload)
    values["codebook"] = CodebookConfig(**values["codebook"])
    values["composite"] = CompositeConfig(**values["composite"])
    return ControlledModelConfig(**values)


def _study_config_from_mapping(payload: dict[str, Any]) -> PopulationGFStudyConfig:
    values = dict(payload)
    values["model_config"] = _model_config_from_mapping(values["model_config"])
    return PopulationGFStudyConfig(**values)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(directory: Path) -> str:
    digest = sha256()
    for name in _REQUIRED_SOURCE_FILES:
        path = directory / name
        digest.update(name.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _load_failed_source(
    source: Path,
) -> tuple[PopulationGFStudyConfig, dict[str, Any], dict[str, Any], str]:
    if not all((source / name).is_file() for name in _REQUIRED_SOURCE_FILES):
        raise ValueError("source population-GF directory is not complete")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((source / "study_config.json").read_text(encoding="utf-8"))
    audit = json.loads((source / "step_halving.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or identity.get("schema_version") != SCHEMA_VERSION
        or audit.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("source population-GF schema is not registered")
    if manifest.get("study_config_hash") != identity.get(
        "study_config_hash"
    ) or manifest.get("study_config_hash") != audit.get("study_config_hash"):
        raise ValueError("source population-GF identities disagree")
    manifest_pass = bool(manifest.get("gf_like_discretization_pass"))
    audit_pass = bool(audit.get("all_registered_parameters_pass"))
    if manifest_pass != audit_pass:
        raise ValueError("source manifest and P38 audit disagree")
    if manifest_pass:
        raise ValueError(
            "source already passed P38; a refinement remedy is not allowed"
        )
    if manifest.get("dynamics") != "euclidean_population_euler":
        raise ValueError("refinement requires a Euclidean population Euler source")
    if tuple(manifest.get("order_parameter_names", ())) != GF_ORDER_PARAMETER_NAMES:
        raise ValueError("source order parameters do not match P37")
    original = _study_config_from_mapping(identity["config"])
    return original, manifest, audit, _source_fingerprint(source)


def _root_is_complete(
    root: Path,
    *,
    config_hash: str,
    source_fingerprint: str,
) -> bool:
    if not (root / "_SUCCESS").is_file():
        return False
    required = (
        "study_config.json",
        "manifest.json",
        "initial_hessian.json",
        "trajectory.json",
        "trajectory.csv",
        "step_halving.json",
    )
    if not all((root / name).is_file() for name in required):
        raise RuntimeError("committed refinement is missing a required artifact")
    identity = json.loads((root / "study_config.json").read_text(encoding="utf-8"))
    if identity.get("study_config_hash") != config_hash:
        raise ValueError("refinement output belongs to another config")
    if (
        identity.get("numerical_refinement", {}).get("source_fingerprint")
        != source_fingerprint
    ):
        raise ValueError("source population-GF artifact changed after refinement")
    return True


def run_population_gf_refinement(
    config: PopulationGFRefinementConfig,
    *,
    output_directory: str | Path,
) -> PopulationGFRefinementResult:
    """Run/resume a nested finer triplet without altering the failed source."""

    source = Path(config.source_directory)
    original, source_manifest, source_audit, source_fingerprint = _load_failed_source(
        source
    )
    original_eta0 = float(source_manifest["eta0"])
    if not math.isfinite(original_eta0) or original_eta0 <= 0.0:
        raise ValueError("source eta0 must be positive and finite")
    factor = config.refinement_factor
    refined_eta0 = original_eta0 / factor
    refined_study = replace(
        original,
        study_id=config.study_id,
        coarse_steps=original.coarse_steps * factor,
        alignment_stride=original.alignment_stride * factor,
    )
    refinement_identity = {
        "config": asdict(refined_study),
        "source_study_config_hash": source_manifest["study_config_hash"],
        "source_fingerprint": source_fingerprint,
        "original_eta0": original_eta0,
        "refinement_factor": factor,
        "refined_eta0": refined_eta0,
    }
    config_hash = canonical_sha256(refinement_identity)
    root = Path(output_directory)
    if _root_is_complete(
        root,
        config_hash=config_hash,
        source_fingerprint=source_fingerprint,
    ):
        rows = json.loads((root / "trajectory.json").read_text(encoding="utf-8"))
        audit = json.loads((root / "step_halving.json").read_text(encoding="utf-8"))
        return PopulationGFRefinementResult(
            root,
            config_hash,
            0,
            3,
            len(rows),
            bool(audit["all_registered_parameters_pass"]),
        )

    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "study_config.json"
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "study_config_hash": config_hash,
        "config": asdict(refined_study),
        "numerical_refinement": refinement_identity,
    }
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing.get("study_config_hash") != config_hash:
            raise ValueError("refinement output directory belongs to another config")
        if (
            existing.get("numerical_refinement", {}).get("source_fingerprint")
            != source_fingerprint
        ):
            raise ValueError("source artifact changed during refinement resume")
    else:
        _write_json(identity_path, identity_payload)

    initial_model = _initialize_model(refined_study)
    initial_state_hash = _model_state_hash(initial_model)
    if initial_state_hash != source_manifest.get("initial_state_hash"):
        raise ValueError("refinement does not reproduce the source initialization")
    population = enumerate_retrieval_population(
        num_concepts=refined_study.model_config.num_concepts,
        memory_size=refined_study.model_config.memory_size,
        dtype=next(initial_model.parameters()).dtype,
        device="cpu",
    )
    by_divisor: dict[int, list[dict[str, Any]]] = {}
    completed = 0
    skipped = 0
    for divisor in (1, 2, 4):
        rows, did_work = _run_or_resume_trajectory(
            config=refined_study,
            config_hash=config_hash,
            population=population,
            root=root,
            divisor=divisor,
            eta0=refined_eta0,
            initial_state_hash=initial_state_hash,
        )
        by_divisor[divisor] = rows
        completed += int(did_work)
        skipped += int(not did_work)

    audit = compute_step_halving_audit(
        {
            divisor: [
                {name: float(row[name]) for name in GF_ORDER_PARAMETER_NAMES}
                for row in rows
            ]
            for divisor, rows in by_divisor.items()
        },
        threshold=refined_study.discrepancy_threshold,
    )
    all_rows = [row for divisor in (1, 2, 4) for row in by_divisor[divisor]]
    _write_json(root / "trajectory.json", all_rows)
    _write_csv(root / "trajectory.csv", all_rows)
    source_hessian = json.loads(
        (source / "initial_hessian.json").read_text(encoding="utf-8")
    )
    _write_json(
        root / "initial_hessian.json",
        {
            **source_hessian,
            "study_config_hash": config_hash,
            "reused_exact_source_hessian": True,
            "source_study_config_hash": source_manifest["study_config_hash"],
        },
    )
    _write_json(
        root / "step_halving.json",
        {
            "schema_version": SCHEMA_VERSION,
            "study_config_hash": config_hash,
            "threshold": audit.threshold,
            "comparisons": audit.comparisons,
            "all_registered_parameters_pass": audit.all_registered_parameters_pass,
            "failed_parameters": list(audit.failed_parameters),
            "gate_type": "deterministic intersection-union finer-Euler remedy",
            "statistical_claim": False,
            "actual_eta_divisors": [factor, 2 * factor, 4 * factor],
        },
    )
    physical_horizon = original.coarse_steps * original_eta0
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": config_hash,
        "dynamics": "euclidean_population_euler",
        "numerical_remedy": True,
        "source_directory": str(source),
        "source_study_config_hash": source_manifest["study_config_hash"],
        "source_fingerprint": source_fingerprint,
        "source_failed_parameters": source_audit.get("failed_parameters", []),
        "original_p36_pass": False,
        "original_eta0": original_eta0,
        "eta0": refined_eta0,
        "eta0_rule": f"numerical remedy: original P36 eta0 / {factor}",
        "refinement_factor": factor,
        "normalized_eta_divisors": [1, 2, 4],
        "actual_eta_divisors": [factor, 2 * factor, 4 * factor],
        "same_physical_horizon": True,
        "physical_horizon": physical_horizon,
        "same_observation_times": True,
        "population_size": population.batch.batch_size,
        "order_parameter_names": list(GF_ORDER_PARAMETER_NAMES),
        "initial_state_hash": initial_state_hash,
        "gf_like_discretization_pass": audit.all_registered_parameters_pass,
        "closure_status": "not_tested",
        "closure_pass": None,
        "closure_claim_eligible": False,
        "interpretation_boundary": (
            "a passing refined triplet resolves a finer numerical reference; it "
            "does not retroactively make the original P36 triplet pass P38"
        ),
        "trajectory_rows": len(all_rows),
        "committed_by": "_SUCCESS written last",
    }
    _write_json(root / "manifest.json", manifest)
    _touch_success(root)
    return PopulationGFRefinementResult(
        root,
        config_hash,
        completed,
        skipped,
        len(all_rows),
        audit.all_registered_parameters_pass,
    )
