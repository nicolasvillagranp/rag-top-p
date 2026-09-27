"""Pipeline entry point: ``python -m topp_rag <config.yaml>``.

Thin glue. Loads the config, then runs the stages in order — each stage is a
no-op when its cache layer already exists, so re-running is cheap and resumable:

    1. Sweep   — embeds (layer 1) + retrieves (layer 2) on a miss, then sweeps
                 the full p × τ × M × k grid (layer 3), all resumable.
    2. Analyze — regenerates every figure and the summary table from the results.

All real logic lives in the package modules; this file only orchestrates.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .analysis import generate_all
from .config import load_config
from .sweep import run_sweep


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m topp_rag <config.yaml>", file=sys.stderr)
        return 2

    config = load_config(argv[0])
    print(f"[topp_rag] config '{config.name}' | datasets={config.datasets} "
          f"| scorer={config.scorer.name} | model={config.embedding.model}")

    print("[topp_rag] running sweep (embed -> retrieve -> grid)...")
    results = run_sweep(config)
    print(f"[topp_rag] sweep produced {len(results)} per-query rows.")

    print(f"[topp_rag] generating analysis into {config.paths.output_dir} ...")
    written = generate_all(results, config.paths.output_dir)
    for path in written:
        print(f"  wrote {Path(path).name}")

    print("[topp_rag] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
