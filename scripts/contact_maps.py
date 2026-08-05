# List mode: python get_contact_maps.py -pdbl ../structures/pdb/1ao7.pdb ../structures/pdb/1bd2.pdb ../structures/pdb/1fo0.pdb  -out ./pdb_cm -workers 8 
# Folder mode: python get_contact_maps.py -pdb ../structures/pdb/ -out ./pdb_cm -workers 8

import os, time, argparse, sys
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio import PDB
from scipy.spatial import cKDTree
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils import parse_general_file, residue_mapping
from config import STRUCTURES_ANNOTATION_DIR


CHAIN_DICT = CHAIN_MAP = OUTPUT_DIR = None

def extract_contacts(pdb_files, chain_dict, distance=5):
    if isinstance(pdb_files, str):
        pdb_files = [pdb_files]

    contacts = []
    parser = PDB.PDBParser(QUIET=True)

    for pdb_file in pdb_files:
        try:
            pdb_id = os.path.basename(pdb_file).replace(".pdb", "")
            structure = parser.get_structure(pdb_id, pdb_file)
            model = structure[0]
            chains = chain_dict.get(pdb_id)

            if not chains:
                print(f"No chain info for {pdb_id}")
                continue

            chain_pairs = [
                (chains['tcra_chain'], chains['mhc_chain']),
                (chains['tcrb_chain'], chains['mhc_chain']),
                (chains['tcra_chain'], chains['peptide_chain']),
                (chains['tcrb_chain'], chains['peptide_chain']),
                (chains['peptide_chain'], chains['mhc_chain']),
            ]

            for cf_id, ct_id in chain_pairs:
                try:
                    cf, ct = model[cf_id], model[ct_id]
                except KeyError:
                    print(f"Chain not found in {pdb_id}: {cf_id} or {ct_id}")
                    continue

                atoms_from = [a for a in cf.get_atoms() if a.get_parent().id[0] == " "]
                atoms_to = [a for a in ct.get_atoms() if a.get_parent().id[0] == " "]

                if not atoms_from or not atoms_to:
                    print(f"No valid atoms in {pdb_id} for {cf_id}-{ct_id}")
                    continue

                tree = cKDTree(np.array([a.coord for a in atoms_to]))

                for atom_from in atoms_from:
                    rf = atom_from.get_parent()
                    if rf.id[0] != " ":
                        continue

                    for idx in tree.query_ball_point(atom_from.coord, distance):
                        atom_to = atoms_to[idx]
                        rt = atom_to.get_parent()
                        if rt.id[0] != " ":
                            continue

                        contacts.append([pdb_id,cf_id,ct_id,
                            residue_mapping.get(rf.get_resname(), rf.get_resname()),
                            residue_mapping.get(rt.get_resname(), rt.get_resname()),
                            rf.id[1],rt.id[1],atom_from.id,atom_to.id,np.linalg.norm(atom_from.coord - atom_to.coord)])

        except Exception as e:
            print(f"{pdb_id}: {e}")
            continue

    return pd.DataFrame(contacts,columns=['pdb_id', 'chain_from', 'chain_to','residue_from', 'residue_to','resid_from', 'resid_to','atom_from', 'atom_to', 'dist'])

def init_worker(chain_dict, chain_map, output_dir):
    global CHAIN_DICT, CHAIN_MAP, OUTPUT_DIR
    CHAIN_DICT = chain_dict
    CHAIN_MAP = chain_map
    OUTPUT_DIR = output_dir

def process_pdb(pdb_path):
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    output_file = os.path.join(OUTPUT_DIR, f"{pdb_id}_contacts.csv")
    if os.path.exists(output_file):
        return "SKIP"

    try:
        df = extract_contacts([pdb_path],{pdb_id: CHAIN_DICT.get(pdb_id, CHAIN_MAP)},distance=10)
        if df.empty:
            return "EMPTY"
        df.to_csv(output_file, index=False)
        return "OK"

    except Exception as e:
        print(f"{pdb_id}: {e}")
        return "ERROR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pdbl", "--pdb_list", nargs="+")
    parser.add_argument("-pdb", "--pdb_folder")
    parser.add_argument("-out", "--output_folder", required=True)
    parser.add_argument("-workers", type=int, default=os.cpu_count())
    parser.add_argument("-cm", "--chain_map", default="D:E:C:B:A")
    parser.add_argument("-notexp", "--not_experimental", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    cm = args.chain_map.split(':')

    chain_map = {
        'tcra_chain': cm[0],
        'tcrb_chain': cm[1],
        'peptide_chain': cm[2],
        'b2m_chain': cm[3],
        'mhc_chain': cm[4]}
    if args.not_experimental:
        chain_dict = chain_map
    else:
        chain_dict = parse_general_file(os.path.join(STRUCTURES_ANNOTATION_DIR, "chain_info.txt"))

    # INPUT MODE
    if args.pdb_list is not None and len(args.pdb_list) > 0:
        pdb_files = list(args.pdb_list)
        print("[INFO] Mode: pdb_list")

    elif args.pdb_folder is not None:
        if not os.path.isdir(args.pdb_folder):
            raise ValueError(f"Invalid folder: {args.pdb_folder}")
        pdb_files = [
            os.path.join(args.pdb_folder, f)
            for f in os.listdir(args.pdb_folder)
            if f.endswith(".pdb")]
        print("[INFO] Mode: pdb_folder")
    else:
        raise ValueError("Provide --pdb_list or --pdb_folder")

    # Skip processed
    pdb_files = [
        f for f in pdb_files
        if not os.path.exists(
            os.path.join(
                args.output_folder,
                f"{os.path.splitext(os.path.basename(f))[0]}_contacts.csv"))]
    
    print("[INFO] Skipping already processed PDBs", len(pdb_files), "remaining")
    workers = min(args.workers, os.cpu_count())
    print(f"[INFO] PDBs: {len(pdb_files)} | Workers: {workers}")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(chain_dict, chain_map, args.output_folder)) as executor:
        results = [f.result()for f in tqdm(as_completed([executor.submit(process_pdb, p) for p in pdb_files]),total=len(pdb_files),desc="Processing")]
    summary = {}
    for status in results:
        summary[status] = summary.get(status, 0) + 1

    print("\n".join(f"{k}: {v}" for k, v in summary.items()))
    print(f"\n[INFO] Done in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()