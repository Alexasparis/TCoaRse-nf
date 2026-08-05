#!/usr/bin/env python3
# python energetic_scorer.py -cmd ../structures/pdb_cm/ ../structures/vdjdb_cm -pot ../potentials/pot -out scores.csv -t 8 -w 8

import argparse
import pandas as pd
import os
import sys
import warnings
import time
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

warnings.simplefilter("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import parse_general_file, filter_contacts
from config import STRUCTURES_ANNOTATION_DIR

# ---------------- GLOBALS ----------------
tcr_p_dict = None
tcr_mhc_dict = None
p_mhc_dict = None

chain_dict = None
chain_map_global = None
threshold_global = None
info_map = None

def init_worker(tcr_p, tcr_mhc, p_mhc, c_dict, c_map, threshold, info_dictionary):
    global tcr_p_dict, tcr_mhc_dict, p_mhc_dict, chain_dict, chain_map_global, threshold_global, info_map

    tcr_p_dict = {
        (r.residue_from, r.residue_to): r.potential
        for r in tcr_p.itertuples(index=False)}

    tcr_mhc_dict = {
        (r.residue_from, r.residue_to): r.potential
        for r in tcr_mhc.itertuples(index=False)}

    p_mhc_dict = {
        (r.residue_from, r.residue_to): r.potential
        for r in p_mhc.itertuples(index=False)}

    chain_dict = c_dict
    chain_map_global = c_map
    threshold_global = threshold
    info_map = info_dictionary


# ---------------- WORKER ----------------
def process_tcr(args_tuple):
    tcr_id, model_num, epitope_seq, pdb_cm_path = args_tuple
    try:
        chains = chain_dict.get(tcr_id, chain_map_global)

        if not os.path.exists(pdb_cm_path):
            return None

        contacts_df = pd.read_csv(pdb_cm_path)
        contacts_TCR_p, contacts_TCR_MHC, contacts_pMHC = filter_contacts(
            contacts_df,
            chains['tcra_chain'],
            chains['tcrb_chain'],
            chains['peptide_chain'],
            chains['mhc_chain'],
            True,
            threshold_global)

        if len(epitope_seq) < 8 or len(epitope_seq) > 13:
            print(f"[WARN] {tcr_id} model={model_num} → Epitope length {len(epitope_seq)} out of expected range (8-13), skipping")
            return None

        # ---------------- TCR-PEPTIDE ----------------
        total_scores = {}
        total_sum = 0

        for i in range(1, len(epitope_seq) + 1):

            sub = contacts_TCR_p[contacts_TCR_p["resid_to"] == i]

            if len(sub) == 0:
                total_scores[f"potential_P{i}"] = 0
                total_scores[f"contacts_P{i}"] = 0
                continue

            pot = [tcr_p_dict.get((a, b), 0)
                for a, b in zip(sub.residue_from, sub.residue_to)]

            total_scores[f"potential_P{i}"] = sum(pot)
            total_scores[f"contacts_P{i}"] = len(sub)
            total_sum += sum(pot)

        # ---------------- TCR-MHC ----------------
        if len(contacts_TCR_MHC) > 0:
            mhc_pot = [
                tcr_mhc_dict.get((a, b), 0)
                for a, b in zip(contacts_TCR_MHC.residue_from, contacts_TCR_MHC.residue_to)]
            mhc_score = sum(mhc_pot)
            mhc_contacts = len(contacts_TCR_MHC)
        else:
            mhc_score = 0
            mhc_contacts = 0

        # ---------------- pMHC ----------------
        total_scores_pmhc = {}
        total_sum_pmhc = 0
        for i in range(1, len(epitope_seq) + 1):
            
            sub = contacts_pMHC[contacts_pMHC["resid_from"] == i]

            if len(sub) == 0:
                total_scores_pmhc[f"potential_pmhc_P{i}"] = 0
                total_scores_pmhc[f"contacts_pmhc_P{i}"] = 0
                continue

            pot = [p_mhc_dict.get((a, b), 0)
                for a, b in zip(sub.residue_from, sub.residue_to)]
            
            total_scores_pmhc[f"potential_pmhc_P{i}"] = sum(pot)
            total_scores_pmhc[f"contacts_pmhc_P{i}"] = len(sub)
            total_sum_pmhc += sum(pot)

        # ---------------- OUTPUT ----------------
        res = {
            "tcr_id": tcr_id,
            "model_number": model_num,
            "e_tcr_p_all": round(total_sum, 4),
            "c_tcr_p_all": sum(total_scores.get(f"contacts_P{i}", 0) for i in range(1, 14)),
            "e_tcr_mhc": round(mhc_score, 4),
            "c_tcr_mhc": mhc_contacts,
            "e_pmhc_all": round(total_sum_pmhc, 4),
            "c_pmhc_all": sum(total_scores_pmhc.get(f"contacts_pmhc_P{i}", 0) for i in range(1, 14))}

        for i in range(1, 14):
            res[f"e_tcr_p{i}"] = round(total_scores.get(f"potential_P{i}", 0), 4)
            res[f"c_tcr_p{i}"] = total_scores.get(f"contacts_P{i}", 0)
            res[f"e_pmhc_p{i}"] = round(total_scores_pmhc.get(f"potential_pmhc_P{i}", 0), 4)
            res[f"c_pmhc_p{i}"] = total_scores_pmhc.get(f"contacts_pmhc_P{i}", 0)

        info = info_map.get(tcr_id)
        if info:
            res.update(info)
        return res

    except Exception as e:
        print(f"[ERROR] {tcr_id}: {e}")
        return None

def chunked(iterable, n):
    it = iter(iterable)
    while chunk := list(itertools.islice(it, n)):
        yield chunk

def extract_chain_sequence(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_file)

    for chain in structure.get_chains():
        if chain.id != chain_id:
            continue

        seq = ""
        for residue in chain:
            if residue.id[0] != " ":
                continue
            try:
                seq += seq1(residue.resname)
            except Exception:
                pass

        return seq

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cmd", "--contact_maps", nargs="+", type=str, required=True)
    parser.add_argument("-pdb", "--pdb_dir", nargs="+", type=str, required=True)
    parser.add_argument("-pot", "--potential_dir", type=str, required=True)
    parser.add_argument("-chains", "--chain_map", type=str, default="D:E:C:A:B")
    parser.add_argument("-out", "--output_file", type=str, default="output.csv")
    parser.add_argument("-t", "--threshold", type=int, required=False, default=7)
    parser.add_argument("-w", "--max_workers", type=int, default=os.cpu_count())
    parser.add_argument("-io", "--io_workers", type=int, default=8)
    parser.add_argument("-notexp", "--not_experimental", action="store_true")
    args = parser.parse_args()
    start_time = time.time()

    print("Loading data...")
    chain_map = dict(zip(["tcra_chain", "tcrb_chain", "peptide_chain", "mhc_chain", "b2m_chain"], args.chain_map.split(":")))
    
    if args.not_experimental:
        chain_dict_local = chain_map
    else:
        chain_dict_local = parse_general_file(os.path.join(STRUCTURES_ANNOTATION_DIR, "chain_info.txt"))
    
    tcr_mhc = pd.read_csv(os.path.join(args.potential_dir, "tcr_mhc_potential.csv"))
    tcr_p = pd.read_csv(os.path.join(args.potential_dir, "tcr_p_potential.csv"))
    p_mhc = pd.read_csv(os.path.join(args.potential_dir, "p_mhc_potential.csv"))

    contact_folders = [c.strip() for c in args.contact_maps]
    info_map_local = {}

    peptide_chain = chain_map["peptide_chain"]

    for pdb_folder in args.pdb_dir:
        if not os.path.isdir(pdb_folder):
            continue

        for f in os.listdir(pdb_folder):
            if not f.endswith(".pdb"):
                continue

            tcr_id = os.path.splitext(f)[0]      # tcr_17_0
            pdb_path = os.path.join(pdb_folder, f)

            epitope = extract_chain_sequence(pdb_path, peptide_chain)

            if epitope is None:
                print(f"[WARN] Could not extract peptide from {f}")
                continue

            info_map_local[tcr_id] = {"Epitope": epitope}

    # ---------------- TASKS ----------------
    tasks = []

    for folder in contact_folders:
        if not os.path.isdir(folder):
            print(f"[WARN] Contact map folder not found: {folder}")
            continue
        for f in os.listdir(folder):
            if not f.endswith("_contacts.csv"):
                continue
            tcr_id = f.replace("_contacts.csv", "")   # tcr_17_0
            try:
                model = int(tcr_id.rsplit("_", 1)[1])
            except Exception:
                model = 0
            ep = info_map_local.get(tcr_id, {}).get("Epitope")
            if ep is None:
                print(f"[WARN] No epitope found for {tcr_id}")
                continue

            tasks.append((tcr_id, model, ep, os.path.join(folder, f)))

    tasks.sort(key=lambda x: (x[0], x[1]))
    print(f"Processing {len(tasks):,} TCRs with {args.max_workers} CPU workers, {args.io_workers} max IO concurrent...")

    # ---------------- PROCESS ----------------
    chunk_size = args.io_workers * 4
    results = []

    with ProcessPoolExecutor(max_workers=args.max_workers,initializer=init_worker,initargs=(tcr_p, tcr_mhc, p_mhc, chain_dict_local, chain_map, args.threshold, info_map_local)) as executor:
        with tqdm(total=len(tasks), desc="Scoring TCRs", unit="tcr",dynamic_ncols=True, colour="cyan") as pbar:
            for chunk in chunked(tasks, chunk_size):
                future_to_task = {executor.submit(process_tcr, t): t for t in chunk}
                pending = set(future_to_task.keys())
                while pending:
                    done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    if not done:
                        pbar.write(f"[WARN] No progress in 30s — {len(pending)} futures hanging")
                        for fut in pending:
                            tcr_id, model, ep, _ = future_to_task[fut]
                            fut.cancel()
                            pbar.write(f"[CANCEL] {tcr_id} model={model}")
                        pending.clear()
                        break

                    for future in done:
                        tcr_id, model, ep, _ = future_to_task[future]
                        try:
                            res = future.result(timeout=5)
                        except Exception as e:
                            pbar.write(f"[ERROR] {tcr_id} model={model} → {e}")
                            pbar.update(1)
                            continue

                        if res is not None:
                            results.append(res)

                        pbar.set_postfix({"tcr": tcr_id, "model": model, "collected": len(results)})
                        pbar.update(1)

    # ---------------- WRITE ----------------
    if results:
        results_df = pd.DataFrame(results)
        results_df["tcr_id"] = results_df["tcr_id"].str.rsplit("_", n=1).str[0] #tcr_17_0 -> tcr_17
        results_df.to_csv(args.output_file, index=False)
        print(f"[WRITE] {len(results):,} rows → {args.output_file}")
    else:
        print("[WARN] no results to write")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
