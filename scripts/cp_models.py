#!/usr/bin/env python3
"""Collect AF3 models into a single PDB folder named {tcr_id}_{model_number}.pdb.

Supported inputs (auto-detected, can even be mixed):
  1) Local AF3 installation:  <input>/<job>/seed-*_sample-*/<job>_seed-*_sample-N_model.cif
                              (or a single job folder: <input>/seed-*_sample-*/...)
  2) AlphaFold Server:        <input>/<prefix>_model_N.cif + <prefix>_job_request.json
                              (or a folder of server downloads: <input>/<any_name>/<prefix>_model_N.cif)
                              The folder/file names do not need to start with 'fold_'.
                              Server jobs are first converted to the local AF3 layout in the
                              staging directory (default: ../results/af3_staging), so the same
                              staged folders can be reused by process_folder.py.
  3) Folder with PDB files:   <input>/tcr_23_X_Y_Z_N.pdb  (last number = model number)

Requires af3_server_adapter.py in the same directory.
"""
import os
import re
import glob
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor
from Bio.PDB import MMCIFParser, PDBIO
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from af3_server_adapter import is_server_job, stage_server_job
from config import DEFAULT_STAGING_DIR

def stage_server_inputs(input_folder: str, staging_dir: str):
    """Convert every AF Server job found in input_folder into the local AF3 layout.

    input_folder can be a single server download or a folder containing several of them.
    Returns the list of staged job folders (staging_dir/<tcr_id>/).
    """
    input_folder = os.path.normpath(input_folder)
    if is_server_job(input_folder):
        candidates = [input_folder]
    else:
        candidates = [os.path.join(input_folder, d) for d in sorted(os.listdir(input_folder))]

    staged = []
    for cand in candidates:
        if os.path.isdir(cand) and is_server_job(cand):
            job_dir = stage_server_job(cand, staging_dir)
            if job_dir:
                staged.append(job_dir)
    return staged


def collect_tasks(input_folder: str, output_folder: str, staging_dir: str):
    """Return a list of (source_file, output_pdb_path) and a count per input type."""
    tasks, counts = {}, {"local": 0, "server": 0, "pdb": 0}

    def add(kind, src, name):
        dst = os.path.join(output_folder, f"{name}.pdb")
        if dst not in tasks:
            tasks[dst] = src
            counts[kind] += 1

    # 1) Local AF3: .../<job>/seed-X_sample-N/*_sample-N_model.cif
    local_cifs = glob.glob(os.path.join(input_folder, "*", "seed-*", "*_model.cif"))
    local_cifs += glob.glob(os.path.join(input_folder, "seed-*", "*_model.cif"))  # single job folder
    for cif in sorted(set(local_cifs)):
        m = re.search(r"sample-(\d+)_model\.cif$", cif)
        if not m:
            continue
        job = os.path.basename(os.path.dirname(os.path.dirname(cif)))
        add("local", cif, f"{job}_{m.group(1)}")

    # 2) AlphaFold Server: stage into the local layout, then collect from the staging dir
    for job_dir in stage_server_inputs(input_folder, staging_dir):
        job = os.path.basename(job_dir)  # job id = server folder name (minus 'fold_' if present)
        for cif in sorted(glob.glob(os.path.join(job_dir, "seed-*", "*_model.cif"))):
            m = re.search(r"sample-(\d+)_model\.cif$", cif)
            if m:
                add("server", cif, f"{job}_{m.group(1)}")

    # 3) Folder with PDB files: tcr_23_X_Y_Z_N.pdb -> tcr_23_N.pdb
    for pdb in sorted(glob.glob(os.path.join(input_folder, "*.pdb"))):
        stem = os.path.splitext(os.path.basename(pdb))[0]
        m = re.search(r"(tcr_\d+).*_(\d+)$", stem)
        if m:
            name = f"{m.group(1)}_{m.group(2)}"
        elif "_" in stem and stem.rsplit("_", 1)[1].isdigit():
            name = stem  # already {id}_{model}
        else:
            print(f"⚠️  Could not parse model number from {os.path.basename(pdb)}, skipped.")
            continue
        add("pdb", pdb, name)

    return [(src, dst) for dst, src in tasks.items()], counts


def convert_or_copy(task):
    src, dst = task
    if os.path.exists(dst):
        return None
    if src.endswith(".cif"):
        structure = MMCIFParser(QUIET=True).get_structure("structure", src)
        io = PDBIO()
        io.set_structure(structure)
        io.save(dst)
    else:
        shutil.copyfile(src, dst)
    return dst


def main():
    parser = argparse.ArgumentParser(
        description="Collect AF3 models (local AF3, AlphaFold Server or PDB folder) into one PDB folder."
    )
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--staging-dir", type=str, default=DEFAULT_STAGING_DIR,
                        help=f"Where AlphaFold Server outputs are converted to the local AF3 layout (default: {DEFAULT_STAGING_DIR}). Only used for AF Server inputs.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    tasks, counts = collect_tasks(args.input_folder, args.output_folder, args.staging_dir)

    if not tasks:
        raise SystemExit(f"No AF3 models found in {args.input_folder}")

    print(f"Found {len(tasks)} models "
          f"(local AF3: {counts['local']}, AF Server: {counts['server']}, PDB: {counts['pdb']}).")
    if counts["server"]:
        print(f"AF Server jobs staged in {args.staging_dir}")
    print(f"Using {args.workers} workers.")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _ in tqdm(executor.map(convert_or_copy, tasks, chunksize=10), total=len(tasks), unit="file"):
            pass

    print("Done.")


if __name__ == "__main__":
    main()