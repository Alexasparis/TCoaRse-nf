#!/usr/bin/env python3
"""Adapter: AlphaFold Server output -> local AF3 layout expected by process_folder.py.

Local AF3 layout (per job):
    <tcr_id>/
        <tcr_id>_data.json
        <tcr_id>_ranking_scores.csv
        seed-<S>_sample-<N>/
            <tcr_id>_seed-<S>_sample-<N>_model.cif
            <tcr_id>_seed-<S>_sample-<N>_confidences.json
            <tcr_id>_seed-<S>_sample-<N>_summary_confidences.json

AlphaFold Server layout (per job, flat; the folder and file prefix can be anything,
e.g. 'fold_tcr_17', 'tcr_17' or 'my_job'):
    <folder>/
        <prefix>_job_request.json
        <prefix>_model_<N>.cif
        <prefix>_full_data_<N>.json
        <prefix>_summary_confidences_<N>.json

Server folders are copied (not modified) into a staging directory with the local layout,
so the original download is left untouched.
"""
import os
import re
import glob
import json
import shutil
from string import ascii_uppercase

import pandas as pd


def is_server_job(folder: str) -> bool:
    """A folder is an AF Server job if it has a *_job_request.json and *_model_<N>.cif files."""
    return (
        os.path.isdir(folder)
        and bool(glob.glob(os.path.join(folder, "*_job_request.json")))
        and bool(glob.glob(os.path.join(folder, "*_model_[0-9]*.cif")))
    )


def is_local_job(folder: str) -> bool:
    """A folder is a local AF3 job if it contains seed-*_sample-* subfolders."""
    return os.path.isdir(folder) and any(
        d.startswith("seed") and os.path.isdir(os.path.join(folder, d))
        for d in os.listdir(folder)
    )


def server_job_id(folder: str) -> str:
    """Job id = folder name. A leading 'fold_' (added by AF Server) is removed if present,
    otherwise the folder name is used as is: 'fold_tcr_17' -> 'tcr_17', 'tcr_17' -> 'tcr_17'.
    Rename the folder if you want a different id."""
    return re.sub(r"^fold_", "", os.path.basename(os.path.abspath(folder)))


def _write_data_json(folder: str, tcr_id: str, dst: str) -> str:
    """Build a local-format <tcr_id>_data.json from the server job_request.json.

    Chains are named A, B, C... following the order of the sequences (counts expanded),
    which is the order AlphaFold Server uses for the chain IDs of the output mmCIF.
    Returns the seed to use in folder names (the server does not expose the seed it used).
    """
    req_path = glob.glob(os.path.join(folder, "*_job_request.json"))[0]
    with open(req_path) as f:
        req = json.load(f)
    job = req[0] if isinstance(req, list) else req

    sequences, idx = [], 0
    for entity in job["sequences"]:
        chain = entity.get("proteinChain")
        if chain is None:
            raise ValueError(f"{req_path}: only proteinChain entities are supported.")
        for _ in range(int(chain.get("count", 1))):
            sequences.append({"protein": {"id": ascii_uppercase[idx], "sequence": chain["sequence"]}})
            idx += 1

    seeds = job.get("modelSeeds") or []
    seed = str(seeds[0]) if seeds else "0"

    data = {
        "name": tcr_id,
        "modelSeeds": [int(seed) if seed.isdigit() else 0],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 2,
    }
    with open(os.path.join(dst, f"{tcr_id}_data.json"), "w") as f:
        json.dump(data, f, indent=2)
    return seed if seed.isdigit() else "0"


def stage_server_job(folder: str, staging_root: str):
    """Copy one AF Server job into staging_root/<tcr_id>/ using the local AF3 layout.

    Returns the staged folder path, or None if no complete model was found.
    """
    tcr_id = server_job_id(folder)
    dst = os.path.join(staging_root, tcr_id)
    os.makedirs(dst, exist_ok=True)
    seed = _write_data_json(folder, tcr_id, dst)

    ranking_rows = []
    for cif in sorted(glob.glob(os.path.join(folder, "*_model_[0-9]*.cif"))):
        m = re.search(r"_model_(\d+)\.cif$", cif)
        if not m:
            continue
        n = m.group(1)
        prefix = cif[: -len(f"_model_{n}.cif")]
        full_json = f"{prefix}_full_data_{n}.json"
        summary_json = f"{prefix}_summary_confidences_{n}.json"
        if not (os.path.exists(full_json) and os.path.exists(summary_json)):
            print(f"⚠️  {tcr_id} model {n}: full_data/summary_confidences missing, skipped.")
            continue

        seed_dir = os.path.join(dst, f"seed-{seed}_sample-{n}")
        os.makedirs(seed_dir, exist_ok=True)
        base = f"{tcr_id}_seed-{seed}_sample-{n}"
        shutil.copy2(cif, os.path.join(seed_dir, f"{base}_model.cif"))
        shutil.copy2(full_json, os.path.join(seed_dir, f"{base}_confidences.json"))
        shutil.copy2(summary_json, os.path.join(seed_dir, f"{base}_summary_confidences.json"))

        with open(summary_json) as f:
            ranking_score = json.load(f).get("ranking_score")
        ranking_rows.append({"seed": seed, "sample": int(n), "ranking_score": ranking_score})

    if not ranking_rows:
        print(f"⚠️  No complete models found in {folder}, skipped.")
        return None

    pd.DataFrame(ranking_rows).to_csv(
        os.path.join(dst, f"{tcr_id}_ranking_scores.csv"), index=False
    )
    return dst


def resolve_jobs(base_path: str, staging_dir: str):
    """Return a list of job folders in local AF3 layout, ready for process_tcr_folders.

    Accepts:
      - a folder containing local AF3 jobs (tcr_17/, tcr_18/, ...)
      - a folder containing unzipped AF Server downloads (any folder name, detected by its
        *_job_request.json and *_model_<N>.cif files)
      - a single AF Server download folder
    """
    base_path = os.path.normpath(base_path)

    if is_server_job(base_path):
        staged = stage_server_job(base_path, staging_dir)
        return [staged] if staged else []

    jobs = []
    for d in sorted(os.listdir(base_path)):
        path = os.path.join(base_path, d)
        if not os.path.isdir(path):
            continue
        if is_server_job(path):
            staged = stage_server_job(path, staging_dir)
            if staged:
                jobs.append(staged)
        elif d.lower().startswith("tcr") or is_local_job(path):
            jobs.append(path)
    return jobs