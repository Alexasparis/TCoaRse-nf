#!/usr/bin/env python3
#python af3_jsons.py --in_df ../out_f/filtered_tcrpmhc.csv --out ../out_f/af3_jsons
import json
import random
import pandas as pd
import argparse
import os

def sample_seed() -> int:
    """Sample a random seed for an AlphaFold3 job."""
    return random.randint(0, 2**32 - 1)

def clean_sequence(seq: str) -> str:
    """Remove '*' characters from a protein sequence."""
    if isinstance(seq, str):
        return seq.replace("*", "")
    return seq

def generate_af3_jsons(tcr_df, b2m="IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM", json_folder="./json_files", non_pep_msa=False):
    os.makedirs(json_folder, exist_ok=True)

    for _, row in tcr_df.iterrows():
        tcr_name = f"TCR_{row['tcr_id']}"
        seed = sample_seed()
        mhc_seq = clean_sequence(row["MHC_seq"])
        epitope_seq = clean_sequence(row["Epitope"])
        tcra_seq = clean_sequence(row["TCRA"])
        tcrb_seq = clean_sequence(row["TCRB"])
        b2m_seq = clean_sequence(b2m)

        sequences = [
            {"protein": {"id": "A", "sequence": mhc_seq}},
            {"protein": {"id": "B", "sequence": b2m_seq}},
            {"protein": {"id": "C", "sequence": epitope_seq}},
            {"protein": {"id": "D", "sequence": tcra_seq}},
            {"protein": {"id": "E", "sequence": tcrb_seq}},
        ]

        if non_pep_msa:
            sequences[2]["protein"]["pairedMsa"] = ""
            sequences[2]["protein"]["unpairedMsa"] = ""

        entry = {
            "name": tcr_name,
            "modelSeeds": [seed],
            "sequences": sequences,
            "dialect": "alphafold3",
            "version": 2,
        }

        file_path = os.path.join(json_folder, f"{tcr_name}.json")
        with open(file_path, "w") as f:
            json.dump(entry, f, indent=4)

    print("✅ AlphaFold3 JSON files successfully generated.")

def main():
    parser = argparse.ArgumentParser(description="Generate AlphaFold3 JSON input files for TCR–pMHC complexes.")
    parser.add_argument("--in_df", required=True, help="Path to the input CSV file containing TCR–pMHC data.")
    parser.add_argument("--out", required=True, help="Path to the output folder where JSON files will be saved.")
    args = parser.parse_args()

    input_csv = args.in_df
    output_folder = args.out

    # Read the input file
    tcr_df = pd.read_csv(input_csv)
    compulsory = ["TCRA", "TCRB", "MHC_seq", "Epitope"]
    if not all(col in tcr_df.columns for col in compulsory):
        missing = [col for col in compulsory if col not in tcr_df.columns]
        raise ValueError(f"Missing compulsory columns: {', '.join(missing)}")

    # If not tcr_id add it as a number
    if "tcr_id" not in tcr_df.columns:
        tcr_df.insert(0, "tcr_id", range(1, len(tcr_df) + 1))

    # Generate AlphaFold3 JSON files
    print(f"Generating AlphaFold3 JSON files in {output_folder}...")
    generate_af3_jsons(tcr_df, json_folder=output_folder)

if __name__ == "__main__":
    main()