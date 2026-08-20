"""Frozen production configurations for the Phase-II controlled matrix.

The builders in this module are deliberately boring: every seed, checkpoint,
evaluation population, architecture, and optimizer choice is visible in one value
object before a production run begins.  They implement Protocol matrices A--D;
finite module localization and exact population GF consume their saved checkpoints
through separate, versioned studies.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from .control_config import (
    CodebookConfig,
    CompositeConfig,
    HeadDesign,
    build_head_capacity_families,
    canonical_sha256,
)
from .controlled_model import ControlledModelConfig
from .controlled_training import ControlledTrainingConfig, ScheduleConfig
from .phase2_study import (
    Phase2CellConfig,
    Phase2StudyConfig,
    run_phase2_study,
)

CHECKPOINT_STEPS = (
    0,
    25,
    50,
    100,
    200,
    400,
    800,
    1200,
    1600,
    2400,
    3200,
    4800,
    6400,
)

COHORT_SEEDS = {
    "discovery-remedy": tuple(range(100, 112)),
    "untouched-confirmation": tuple(range(1000, 1024)),
    "optimizer-replication": tuple(range(2000, 2016)),
}

# Differences exceed every cohort's full seed range, and Phase2StudyConfig also
# checks the realized Cartesian set for collisions.
STREAM_OFFSETS = {
    "init_seed_offset": 10_000_000,
    "train_seed_offset": 20_000_000,
    "eval_seed_offset": 30_000_000,
    "walsh_seed_offset": 40_000_000,
    "swap_seed_offset": 50_000_000,
    "patch_seed_offset": 60_000_000,
    "diag_seed_offset": 70_000_000,
}

_RANDOM_CODEBOOK_SALT = 271_801
_LOW_COHERENCE_REPLICAS = (1701, 1702, 1703, 1704)


def _optimizer_for_cohort(cohort: str) -> tuple[str, float]:
    """Use AdamW for headline cohorts and momentum SGD for replication."""

    if cohort == "optimizer-replication":
        return "sgd", 0.9
    return "adamw", 0.0


def _model(
    *,
    num_concepts: int = 32,
    num_heads: int = 4,
    attention_width: int = 8,
    ffn_width: int | None,
    composite: str = "factorized",
    geometry: str = "random_normalized",
    codebook_trainable: bool = True,
    codebook_seed: int = _RANDOM_CODEBOOK_SALT,
) -> ControlledModelConfig:
    return ControlledModelConfig(
        memory_size=4,
        num_layers=2,
        num_heads=num_heads,
        attention_width=attention_width,
        beta=1.0,
        ffn_width=ffn_width,
        codebook=CodebookConfig(
            num_concepts=num_concepts,
            d_model=8,
            geometry=geometry,
            trainable=codebook_trainable,
            seed=codebook_seed,
        ),
        composite=CompositeConfig(kind=composite),
    )


def _training(
    *,
    cohort: str,
    schedule_kind: str,
    end_step: int,
    branch_step: int,
) -> ControlledTrainingConfig:
    optimizer, momentum = _optimizer_for_cohort(cohort)
    return ControlledTrainingConfig(
        batch_size=256,
        optimizer=optimizer,
        momentum=momentum,
        weight_decay=0.0,
        schedule=ScheduleConfig(
            kind=schedule_kind,
            base_learning_rate=0.003,
            branch_step=branch_step,
            end_step=end_step,
        ),
    )


def _cell(
    *,
    cohort: str,
    arm_name: str,
    model_config: ControlledModelConfig,
    schedule_kind: str = "constant",
    end_step: int = 6400,
    branch_step: int = 0,
    codebook_seed_policy: str = "master_init",
    codebook_replica_seeds: tuple[int, ...] = (),
) -> Phase2CellConfig:
    checkpoints = tuple(step for step in CHECKPOINT_STEPS if step <= end_step)
    return Phase2CellConfig(
        arm_name=arm_name,
        model_config=model_config,
        training_config=_training(
            cohort=cohort,
            schedule_kind=schedule_kind,
            end_step=end_step,
            branch_step=branch_step,
        ),
        checkpoint_steps=checkpoints,
        codebook_seed_policy=codebook_seed_policy,
        codebook_replica_seeds=codebook_replica_seeds,
    )


def _study(
    *,
    cohort: str,
    name: str,
    cells: tuple[Phase2CellConfig, ...],
) -> Phase2StudyConfig:
    if cohort not in COHORT_SEEDS:
        raise ValueError(f"unknown Phase-II cohort {cohort!r}")
    return Phase2StudyConfig(
        study_id=f"phase2-{name}-{cohort}-v2",
        cohort=cohort,
        cells=cells,
        seeds=COHORT_SEEDS[cohort],
        evaluation_batch_size=8192,
        walsh_skeleton_count=512,
        swap_pair_count=2048,
        **STREAM_OFFSETS,
    )


def _residual_factorization_study(
    *,
    cohort: str,
    ffn_width: int | None,
) -> Phase2StudyConfig:
    """Combine matrices A/B so the baseline trajectory is not trained twice."""

    hard_factorized = _model(ffn_width=ffn_width)
    hard_rank = replace(
        hard_factorized,
        composite=CompositeConfig(kind="rank_matched_direct"),
    )
    hard_dense = replace(
        hard_factorized,
        composite=CompositeConfig(kind="dense_direct"),
    )
    h1_factorized = replace(hard_factorized, num_heads=1, attention_width=8)
    h1_dense = replace(
        h1_factorized,
        composite=CompositeConfig(kind="dense_direct"),
    )
    cells = (
        _cell(
            cohort=cohort,
            arm_name="hard-factorized-constant-6400",
            model_config=hard_factorized,
            branch_step=800,
        ),
        _cell(
            cohort=cohort,
            arm_name="hard-factorized-cosine-3200",
            model_config=hard_factorized,
            schedule_kind="cosine",
            end_step=3200,
            branch_step=800,
        ),
        _cell(
            cohort=cohort,
            arm_name="hard-factorized-cosine-6400",
            model_config=hard_factorized,
            schedule_kind="cosine",
            branch_step=800,
        ),
        _cell(
            cohort=cohort,
            arm_name="hard-rank-matched-constant-6400",
            model_config=hard_rank,
            branch_step=800,
        ),
        _cell(
            cohort=cohort,
            arm_name="hard-dense-direct-constant-6400",
            model_config=hard_dense,
            branch_step=800,
        ),
        _cell(
            cohort=cohort,
            arm_name="h1-factorized-constant-6400",
            model_config=h1_factorized,
            branch_step=800,
        ),
        _cell(
            cohort=cohort,
            arm_name="h1-dense-direct-constant-6400",
            model_config=h1_dense,
            branch_step=800,
        ),
    )
    suffix = "noffn" if ffn_width is None else "ffn"
    return _study(
        cohort=cohort,
        name=f"residual-factorization-{suffix}",
        cells=cells,
    )


def _representation_study(*, cohort: str) -> Phase2StudyConfig:
    random_learned = _model(ffn_width=None)
    random_fixed = replace(
        random_learned,
        codebook=replace(random_learned.codebook, trainable=False),
    )
    low_fixed = _model(
        ffn_width=None,
        geometry="low_coherence",
        codebook_trainable=False,
        codebook_seed=_LOW_COHERENCE_REPLICAS[0],
    )
    low_learned = replace(
        low_fixed,
        codebook=replace(low_fixed.codebook, trainable=True),
    )
    orthogonal = _model(
        num_concepts=8,
        ffn_width=None,
        geometry="orthogonal",
        codebook_trainable=False,
        codebook_seed=271_802,
    )
    cells = (
        _cell(
            cohort=cohort,
            arm_name="random-learned",
            model_config=random_learned,
        ),
        _cell(
            cohort=cohort,
            arm_name="random-fixed",
            model_config=random_fixed,
        ),
        _cell(
            cohort=cohort,
            arm_name="low-coherence-learned",
            model_config=low_learned,
            codebook_seed_policy="balanced_replicas",
            codebook_replica_seeds=_LOW_COHERENCE_REPLICAS,
        ),
        _cell(
            cohort=cohort,
            arm_name="low-coherence-fixed",
            model_config=low_fixed,
            codebook_seed_policy="balanced_replicas",
            codebook_replica_seeds=_LOW_COHERENCE_REPLICAS,
        ),
        _cell(
            cohort=cohort,
            arm_name="orthogonal-c8-fixed-negative-control",
            model_config=orthogonal,
        ),
    )
    return _study(cohort=cohort, name="representation-source", cells=cells)


def _head_model(design: HeadDesign) -> ControlledModelConfig:
    return _model(
        num_heads=design.num_heads,
        attention_width=design.attention_width,
        ffn_width=design.ffn_width,
    )


def _head_capacity_study(*, cohort: str) -> Phase2StudyConfig:
    families = build_head_capacity_families(
        d_model=8,
        head_counts=(1, 2, 4, 8),
    )
    cells = tuple(
        _cell(
            cohort=cohort,
            arm_name=f"{family}-h{design.num_heads}",
            model_config=_head_model(design),
        )
        for family, designs in families.items()
        for design in designs
    )
    return _study(cohort=cohort, name="head-capacity", cells=cells)


def build_phase2_studies(*, cohort: str) -> dict[str, Phase2StudyConfig]:
    """Return all preregistered architecture studies for one seed cohort."""

    return {
        "residual-factorization-noffn": _residual_factorization_study(
            cohort=cohort,
            ffn_width=None,
        ),
        "residual-factorization-ffn": _residual_factorization_study(
            cohort=cohort,
            ffn_width=16,
        ),
        "representation-source": _representation_study(cohort=cohort),
        "head-capacity": _head_capacity_study(cohort=cohort),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    """Write strict canonical presentation bytes and publish them atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_phase2_config_bundle(
    *,
    cohort: str,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Freeze the four exact study configs before trusting any training output.

    The dataclass-to-JSON representation is human-readable, while
    ``study_config_hash`` uses the same canonical identity as the execution planner.
    The small index makes it possible to audit a published directory without first
    importing the Python package.
    """

    root = Path(output_directory)
    studies = build_phase2_studies(cohort=cohort)
    paths: list[Path] = []
    index: dict[str, object] = {
        "schema_version": "phase2-config-bundle-v1",
        "cohort": cohort,
        "studies": {},
    }
    index_studies = index["studies"]
    assert isinstance(index_studies, dict)
    for name, config in studies.items():
        config_hash = canonical_sha256(config)
        path = root / f"{name}.json"
        _write_json_atomic(
            path,
            {
                "schema_version": "phase2-config-bundle-v1",
                "study_name": name,
                "study_config_hash": config_hash,
                "config": asdict(config),
            },
        )
        paths.append(path)
        index_studies[name] = {
            "filename": path.name,
            "study_config_hash": config_hash,
        }
    index_path = root / "index.json"
    _write_json_atomic(index_path, index)
    paths.append(index_path)
    return tuple(paths)


def main(argv: list[str] | None = None) -> None:
    """Run one named frozen study; the runner writes the full config manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=tuple(COHORT_SEEDS), required=True)
    parser.add_argument(
        "--study",
        choices=(
            "residual-factorization-noffn",
            "residual-factorization-ffn",
            "representation-source",
            "head-capacity",
        ),
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config = build_phase2_studies(cohort=args.cohort)[args.study]
    # ``asdict`` is intentionally evaluated before execution: malformed dataclass
    # contents fail before a result directory can be mistaken for a valid study.
    asdict(config)
    summary = run_phase2_study(
        config=config,
        output_directory=args.output_directory,
        device=args.device,
    )
    print(asdict(summary))


if __name__ == "__main__":
    main()
