import os
import re
import uuid
import shutil
import tempfile
import argparse
import pandas as pd
import subprocess
from itertools import combinations
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def remove_headers(file_path):
    with open(file_path) as f:
        return [l for l in f if l.startswith("ATOM") and len(l) > 21]

def merge_pdb(pdb_file, chain_mapping={"tcra_chain":"D","tcrb_chain":"E","mhc_chain":"A","b2_chain":"B","peptide_chain":"C"}):
    tmpdir = tempfile.mkdtemp(prefix=f"{uuid.uuid4().hex[:8]}_")

    cleaned_dir = os.path.join(tmpdir, "cleaned")
    merged_dir = os.path.join(tmpdir, "merged")
    os.makedirs(cleaned_dir)
    os.makedirs(merged_dir)

    base = os.path.splitext(os.path.basename(pdb_file))[0]

    cleaned = os.path.join(cleaned_dir, f"{base}.pdb")
    receptor = os.path.join(cleaned_dir, "A.pdb")
    ligand = os.path.join(cleaned_dir, "B.pdb")
    merged = os.path.join(merged_dir, f"{base}_merged.pdb")

    with open(cleaned, "w") as f:
        f.writelines(remove_headers(pdb_file))

    subprocess.run(f"pdb_selchain -{chain_mapping['mhc_chain']},{chain_mapping['b2_chain']},{chain_mapping['peptide_chain']} {cleaned} | pdb_chain -A | pdb_reres -1 | pdb_delhetatm > {receptor}", shell=True, check=True)
    subprocess.run(f"pdb_selchain -{chain_mapping['tcra_chain']},{chain_mapping['tcrb_chain']} {cleaned} | pdb_chain -B | pdb_reres -1 | pdb_delhetatm > {ligand}", shell=True, check=True)

    with open(merged, "w") as f:
        f.writelines(remove_headers(receptor) + remove_headers(ligand))

    return merged, tmpdir

def run_dockq(model, native):
    out = subprocess.run(f"DockQ {model} {native} --mapping AB:AB", shell=True, capture_output=True, text=True, check=True).stdout
    grab = lambda p, t=float: t(re.search(p, out).group(1)) if re.search(p, out) else None
    return grab(r"DockQ:\s*([\d\.]+)"), grab(r"iRMSD:\s*([\d\.]+)"), grab(r"LRMSD:\s*([\d\.]+)"), grab(r"fnat:\s*([\d\.]+)"), grab(r"clashes:\s*(\d+)", int)

def generate_dict(folder):
    d = {}
    for f in os.listdir(folder):
        if f.endswith(".pdb") and "_merged" not in f:
            tcr_id, model = os.path.splitext(f)[0].rsplit("_", 1)
            d.setdefault(tcr_id, {})[int(model)] = os.path.join(folder, f)
    return d

def process_job(job):
    tcr_id, model_i, model_j, pdb_i, pdb_j = job
    pdb_i, tmp_i = merge_pdb(pdb_i)
    pdb_j, tmp_j = merge_pdb(pdb_j)
    try:
        metrics = run_dockq(pdb_i, pdb_j)
    finally:
        shutil.rmtree(tmp_i, ignore_errors=True)
        shutil.rmtree(tmp_j, ignore_errors=True)
    return (tcr_id, model_i, model_j, *metrics)

def process_comparisons(data, workers):
    jobs = [(tcr_id, i, j, models[i], models[j]) for tcr_id, models in data.items() for i, j in combinations(sorted(models), 2)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_job, job) for job in jobs]
        results = [f.result() for f in tqdm(as_completed(futures), total=len(jobs), desc="DockQ")]
    return pd.DataFrame(results, columns=["tcr_id","model_i","model_j","DockQ","iRMSD","LRMSD","fnat","clashes"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    process_comparisons(generate_dict(args.folder), args.workers).to_csv(args.output, index=False)

if __name__ == "__main__":
    main()