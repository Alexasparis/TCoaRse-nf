#!/usr/bin/env python3
"""Process one manifest chunk on one SLURM node."""

from __future__ import annotations

import argparse
import fnmatch
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import traceback

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required: conda install pyyaml") from exc


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def read_chunk(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header != ["batch", "name", "pdb_path"]:
            raise SystemExit(f"Unexpected chunk header in {path}: {header}")
        rows = []
        for line in fh:
            if not line.strip():
                continue
            batch, name, pdb_path = line.rstrip("\n").split("\t")
            rows.append({"batch": batch, "name": name, "pdb_path": pdb_path})
    return rows


def load_chain_mapping(cfg: dict) -> dict | None:
    mapping_path = cfg.get("chain_mapping_json")
    if not mapping_path:
        return None
    with Path(mapping_path).open() as fh:
        mapping = json.load(fh)
    if not isinstance(mapping, dict):
        raise SystemExit(f"chain_mapping_json must contain a JSON object: {mapping_path}")
    return mapping


def chain_mapping_key(name: str, cfg: dict) -> str:
    mode = cfg.get("chain_mapping_key_mode", "exact")
    if mode == "exact":
        return name
    if mode == "prefix_before_underscore":
        return name.split("_", 1)[0]
    if mode == "first4":
        return name[:4]
    raise ValueError(f"Unsupported chain_mapping_key_mode: {mode}")


def pydock_modules(cfg: dict) -> list[str]:
    modules = cfg.get("pydock_modules", "bindEy")
    if isinstance(modules, str):
        modules = [
            part
            for part in re.split(r"[\s,]+", modules.strip())
            if part
        ]
    if not isinstance(modules, list) or not modules:
        raise SystemExit("pydock_modules must be a non-empty string or list")
    for module in modules:
        if not isinstance(module, str) or not module.strip():
            raise SystemExit(f"Invalid pydock module entry: {module!r}")
    return [module.strip() for module in modules]


def keep_file_patterns(cfg: dict) -> list[str]:
    patterns = cfg.get("keep_file_patterns", ["*.ene"])
    if isinstance(patterns, str):
        patterns = [
            part
            for part in re.split(r"[\s,]+", patterns.strip())
            if part
        ]
    if not isinstance(patterns, list) or not patterns:
        raise SystemExit("keep_file_patterns must not be empty")
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.strip():
            raise SystemExit(f"Invalid keep_file_patterns entry: {pattern!r}")
    return [pattern.strip() for pattern in patterns]


def should_keep_file(path: Path, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def write_ini(
    path: Path,
    name: str,
    cfg: dict,
    chain_mapping: dict | None,
) -> None:
    if chain_mapping is None:
        receptor_mol = "A,B,C"
        ligand_mol = "D,E"
    else:
        key = chain_mapping_key(name, cfg)
        try:
            chains = chain_mapping[key]
            receptor_mol = ",".join([
                chains["peptide_chain"],
                chains["mhc_chain"],
                chains["b2_chain"],
            ])
            ligand_mol = ",".join([
                chains["tcra_chain"],
                chains["tcrb_chain"],
            ])
        except KeyError as exc:
            raise ValueError(
                f"Missing chain mapping for {name} using key {key}: {exc}"
            ) from exc

    path.write_text(
        "[receptor]\n"
        f"pdb = {name}.pdb\n"
        f"mol = {receptor_mol}\n"
        "newmol = A\n"
        "\n"
        "[ligand]\n"
        f"pdb = {name}.pdb\n"
        f"mol = {ligand_mol}\n"
        "newmol = D\n"
    )


def run_one(
    row: dict[str, str],
    local_sif: str,
    local_results_root: str,
    cfg: dict,
    chain_mapping: dict | None,
) -> tuple[bool, str, str, str]:
    batch = row["batch"]
    name = row["name"]
    pdb_path = Path(row["pdb_path"])
    workdir = Path(local_results_root) / batch / name
    workdir.mkdir(parents=True, exist_ok=True)
    patterns = keep_file_patterns(cfg)

    try:
        local_pdb = workdir / f"{name}.pdb"
        if cfg.get("copy_pdb_to_tmp", True):
            shutil.copy2(pdb_path, local_pdb)
        else:
            if not local_pdb.exists():
                os.symlink(pdb_path, local_pdb)

        write_ini(workdir / f"{name}.ini", name, cfg, chain_mapping)

        stdout = (
            subprocess.DEVNULL
            if cfg.get("suppress_pydock_logs", True)
            else None
        )
        stderr = (
            subprocess.DEVNULL
            if cfg.get("suppress_pydock_logs", True)
            else None
        )

        cmd_prefix = [
            "singularity",
            cfg.get("singularity_subcommand", "run"),
            local_sif,
        ]
        entrypoint = cfg.get("container_entrypoint", "")
        if entrypoint:
            cmd_prefix.append(entrypoint)

        for module in pydock_modules(cfg):
            cmd = cmd_prefix + [name, module]
            subprocess.run(
                cmd,
                cwd=workdir,
                stdout=stdout,
                stderr=stderr,
                check=True,
            )

        ene = workdir / f"{name}.ene"
        if not ene.exists():
            return (
                False,
                batch,
                name,
                "pyDock completed but .ene file is missing",
            )

        for entry in workdir.iterdir():
            if should_keep_file(entry, patterns):
                continue
            if entry.is_file() or entry.is_symlink():
                entry.unlink()
        return True, batch, name, ""
    except Exception as exc:
        return (
            False,
            batch,
            name,
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def make_batch_shards(
    local_results_root: Path,
    output_dir: Path,
    chunk_id: int,
    cfg: dict,
) -> None:
    shards_root = output_dir / "archives" / "shards"
    patterns = keep_file_patterns(cfg)
    for batch_dir in sorted(
        p for p in local_results_root.iterdir() if p.is_dir()
    ):
        batch = batch_dir.name
        shard_dir = shards_root / batch
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"chunk_{chunk_id:06d}.tar"
        tmp_shard = shard_path.with_suffix(".tar.tmp")
        with tarfile.open(tmp_shard, "w") as tar:
            for file_path in sorted(batch_dir.glob("*/*")):
                if not file_path.is_file() and not file_path.is_symlink():
                    continue
                if not should_keep_file(file_path, patterns):
                    continue
                arcname = f"{file_path.parent.name}/{file_path.name}"
                tar.add(file_path, arcname=arcname)
        tmp_shard.replace(shard_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--chunk", required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--local-sif", required=True)
    parser.add_argument("--cpus", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    output_dir = Path(cfg["output_dir"]).resolve()
    chunk_path = Path(args.chunk).resolve()
    cpus = args.cpus or int(os.environ.get("SLURM_CPUS_PER_TASK", cfg.get("cpus_per_task", 1)))
    rows = read_chunk(chunk_path)
    chain_mapping = load_chain_mapping(cfg)

    tmp_base = Path(os.environ.get("TMPDIR", tempfile.gettempdir())).resolve()
    local_results_root = tmp_base / f"pydock_array_{os.environ.get('SLURM_JOB_ID', 'local')}_{args.chunk_id:06d}"
    local_results_root.mkdir(parents=True, exist_ok=True)

    failures = []
    done = 0
    with ProcessPoolExecutor(max_workers=cpus) as pool:
        futures = [pool.submit(run_one, row, args.local_sif, str(local_results_root), cfg, chain_mapping) for row in rows]
        for fut in as_completed(futures):
            ok, batch, name, message = fut.result()
            done += 1
            if not ok:
                failures.append((batch, name, message.replace("\n", "\\n")))
            if done % 100 == 0 or done == len(rows):
                print(f"chunk {args.chunk_id:06d}: {done}/{len(rows)} complexes complete", flush=True)

    make_batch_shards(local_results_root, output_dir, args.chunk_id, cfg)

    failures_dir = output_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    failures_path = failures_dir / f"chunk_{args.chunk_id:06d}.failures.tsv"
    with failures_path.open("w") as out:
        out.write("batch\tname\terror\n")
        for batch, name, message in failures:
            out.write(f"{batch}\t{name}\t{message}\n")

    print(f"chunk {args.chunk_id:06d}: {len(rows) - len(failures)} succeeded, {len(failures)} failed")
    if failures:
        print(f"Failures written to {failures_path}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
