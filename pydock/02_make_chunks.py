#!/usr/bin/env python3
"""Split manifest.tsv into chunk_XXXXXX.tsv files for SLURM arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required: conda install pyyaml") from exc


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--chunks-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    output_dir = Path(cfg["output_dir"]).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else output_dir / "manifests" / "manifest.tsv"
    chunks_dir = Path(args.chunks_dir).resolve() if args.chunks_dir else output_dir / "chunks"
    chunk_size = args.chunk_size or int(cfg.get("complexes_per_chunk", 5000))
    if chunk_size <= 0:
        raise SystemExit("chunk size must be positive")

    chunks_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open() as fh:
        header = fh.readline()
        if header.rstrip("\n") != "batch\tname\tpdb_path":
            raise SystemExit(f"Unexpected manifest header: {header!r}")
        rows = [line for line in fh if line.strip()]

    if not rows:
        raise SystemExit("Manifest contains no rows")

    n_chunks = 0
    for start in range(0, len(rows), chunk_size):
        chunk_path = chunks_dir / f"chunk_{n_chunks:06d}.tsv"
        with chunk_path.open("w") as out:
            out.write(header)
            out.writelines(rows[start:start + chunk_size])
        n_chunks += 1

    index_path = chunks_dir / "chunks.index"
    with index_path.open("w") as out:
        for i in range(n_chunks):
            out.write(f"{i}\t{chunks_dir / f'chunk_{i:06d}.tsv'}\n")

    print(f"Wrote {n_chunks} chunks to {chunks_dir}", file=sys.stderr)
    print(f"Submit with: sbatch --array=0-{n_chunks - 1} run_array.sh", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
