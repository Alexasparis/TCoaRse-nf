import os
import re
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor
from Bio.PDB import MMCIFParser, PDBIO
from tqdm import tqdm
from itertools import repeat

def cif_to_pdb_og(args):
    cif_file, output_folder = args
    match = re.search(r"tcr_(\d+).*sample-(\d+)_model\.cif$", cif_file)
    if not match:
        return None
    tcr_id, sample_id = match.group(1), match.group(2)
    output_pdb = os.path.join(output_folder, f"{tcr_id}_{sample_id}.pdb")
    if os.path.exists(output_pdb):
        return None
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    return output_pdb
    
def cif_to_pdb(args):
    cif_file, output_folder = args
    folder_name = os.path.basename(os.path.dirname(os.path.dirname(cif_file)))
    match = re.search(r"sample-(\d+)_model\.cif$", cif_file)
    if not match:
        return None

    sample_id = match.group(1)
    output_pdb = os.path.join(output_folder, f"{folder_name}_{sample_id}.pdb")

    if os.path.exists(output_pdb):
        return None

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)

    return output_pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    cif_files = glob.glob(os.path.join(args.input_folder, "*", "seed-*", "*_model.cif"))

    print(f"Found {len(cif_files)} CIF files.")
    print(f"Using {args.workers} workers.")
    tasks = zip(cif_files, repeat(args.output_folder))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _ in tqdm(executor.map(cif_to_pdb, tasks, chunksize=10),total=len(cif_files),unit="file",):
            pass

    print("Done.")

if __name__ == "__main__":
    main()
