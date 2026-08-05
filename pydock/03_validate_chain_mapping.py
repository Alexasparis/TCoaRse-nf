#!/usr/bin/env python3
"""Validate that all manifest entries have the configured chain mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required: conda install pyyaml") from exc

REQUIRED_KEYS = ["tcra_chain", "tcrb_chain", "peptide_chain", "mhc_chain", "b2_chain"]


def mapping_key(name: str, mode: str) -> str:
    if mode == "exact":
        return name
    if mode == "prefix_before_underscore":
        return name.split("_", 1)[0]
    if mode == "first4":
        return name[:4]
    raise SystemExit(f"Unsupported chain_mapping_key_mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mapping_path = cfg.get("chain_mapping_json")
    if not mapping_path:
        print("chain_mapping_json is not set; fixed-chain mode will be used.")
        return 0

    output_dir = Path(cfg["output_dir"]).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else output_dir / "manifests" / "manifest.tsv"
    mapping = json.loads(Path(mapping_path).read_text())
    mode = cfg.get("chain_mapping_key_mode", "exact")

    missing = []
    incomplete = []
    with manifest.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header != ["batch", "name", "pdb_path"]:
            raise SystemExit(f"Unexpected manifest header: {header}")
        for line in fh:
            if not line.strip():
                continue
            batch, name, pdb_path = line.rstrip("\n").split("\t")
            key = mapping_key(name, mode)
            chains = mapping.get(key)
            if chains is None:
                missing.append((batch, name, key))
                continue
            absent = [field for field in REQUIRED_KEYS if field not in chains or not chains[field]]
            if absent:
                incomplete.append((batch, name, key, ",".join(absent)))

    if missing or incomplete:
        print(f"Missing mappings: {len(missing)}")
        for batch, name, key in missing[:20]:
            print(f"  missing: batch={batch} name={name} key={key}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")
        print(f"Incomplete mappings: {len(incomplete)}")
        for batch, name, key, absent in incomplete[:20]:
            print(f"  incomplete: batch={batch} name={name} key={key} missing_fields={absent}")
        if len(incomplete) > 20:
            print(f"  ... {len(incomplete) - 20} more")
        return 1

    print(f"All manifest entries have complete chain mappings using mode '{mode}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
