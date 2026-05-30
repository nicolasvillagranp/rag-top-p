"""Configuration loading — YAML into a typed Config object.

Every module reads the same schema through this loader so the pipeline has a
single source of truth for hyperparameters and paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    batch_size: int = 64


@dataclass(frozen=True)
class RetrievalConfig:
    M_values: list[int]
    M_max: int


@dataclass(frozen=True)
class ScorerConfig:
    """Which similarity function produces the score arrays.

    ``name`` is threaded into cache filenames so different scorers never
    collide. ``params`` carries scorer-specific knobs (e.g. ``alpha`` for the
    future hybrid scorer).
    """

    name: str = "cosine"
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TopPConfig:
    p_values: list[float]
    tau_values: list[float]
    k_values: list[int]


@dataclass(frozen=True)
class EvaluationConfig:
    max_queries: int | None = None


@dataclass(frozen=True)
class PathsConfig:
    cache_dir: Path
    output_dir: Path
    fixtures_dir: Path

    @property
    def embeddings_dir(self) -> Path:
        return self.cache_dir / "embeddings"

    @property
    def retrievals_dir(self) -> Path:
        return self.cache_dir / "retrievals"

    @property
    def results_dir(self) -> Path:
        return self.cache_dir / "results"


@dataclass(frozen=True)
class Config:
    name: str
    seed: int
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    scorer: ScorerConfig
    top_p: TopPConfig
    datasets: list[str]
    evaluation: EvaluationConfig
    paths: PathsConfig


def load_config(path: str | Path) -> Config:
    """Parse a YAML config file into a :class:`Config`.

    The ``scorer`` block is optional and defaults to cosine, keeping older
    configs (which predate the scorer abstraction) valid.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    experiment = raw.get("experiment", {})
    scorer_raw = raw.get("scorer", {}) or {}

    return Config(
        name=experiment.get("name", path.stem),
        seed=int(experiment.get("seed", 42)),
        embedding=EmbeddingConfig(
            model=raw["embedding"]["model"],
            batch_size=int(raw["embedding"].get("batch_size", 64)),
        ),
        retrieval=RetrievalConfig(
            M_values=list(raw["retrieval"]["M_values"]),
            M_max=int(raw["retrieval"]["M_max"]),
        ),
        scorer=ScorerConfig(
            name=scorer_raw.get("name", "cosine"),
            params=dict(scorer_raw.get("params", {})),
        ),
        top_p=TopPConfig(
            p_values=list(raw["top_p"]["p_values"]),
            tau_values=list(raw["top_p"]["tau_values"]),
            k_values=list(raw["top_p"]["k_values"]),
        ),
        datasets=list(raw["datasets"]),
        evaluation=EvaluationConfig(
            max_queries=raw.get("evaluation", {}).get("max_queries"),
        ),
        paths=PathsConfig(
            cache_dir=Path(raw["paths"]["cache_dir"]),
            output_dir=Path(raw["paths"]["output_dir"]),
            fixtures_dir=Path(raw["paths"]["fixtures_dir"]),
        ),
    )