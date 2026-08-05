#!/usr/bin/env python3
"""Build a stable manifest of all input PDB files.

Output columns:
    batch<TAB>name<TAB>pdb_path
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required: conda install pyyaml") from exc


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def iter_pdbs(input_dir: Path):
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".pdb"):
                yield entry.name, Path(entry.path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    output_dir = Path(cfg["output_dir"]).resolve()
    manifest = Path(args.output).resolve() if args.output else output_dir / "manifests" / "manifest.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    count = 0
    with manifest.open("w") as out:
        out.write("batch\tname\tpdb_path\n")
        for raw_input_dir in cfg["input_dirs"]:
            input_dir = Path(raw_input_dir).resolve()
            if not input_dir.is_dir():
                raise SystemExit(f"Input directory does not exist: {input_dir}")
            batch = input_dir.name
            batch_count = 0
            for filename, pdb_path in sorted(iter_pdbs(input_dir)):
                name = pdb_path.stem
                key = (batch, name)
                if key in seen:
                    raise SystemExit(f"Duplicate complex in batch {batch}: {name}")
                seen.add(key)
                out.write(f"{batch}\t{name}\t{pdb_path}\n")
                count += 1
                batch_count += 1
            print(f"{batch}: {batch_count} PDB files", file=sys.stderr)

    if count == 0:
        raise SystemExit("No .pdb files found")
    print(f"Wrote {count} entries to {manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
