import random
import json
import subprocess
import pandas as pd
import argparse
import os

def reorder_add_columns(df):
    df_reordered = df.copy()
    cols = ["TRAV","TRAJ","TRAC","TRBC","TRA_leader","TRB_leader","Linker","Link_order",
            "TRA_5_prime_seq","TRA_3_prime_seq","TRB_5_prime_seq","TRB_3_prime_seq"]
    for c in cols: 
        if c not in df_reordered: df_reordered[c] = ""
    df_reordered = df_reordered.rename(columns={"tcr_id": "TCR_name", "CDR3a": "TRA_CDR3", "CDR3b": "TRB_CDR3"})
    df_reordered = df_reordered[["TCR_name","TRAV","TRAJ","TRA_CDR3","TRBV","TRBJ","TRB_CDR3",
               "TRAC","TRBC","TRA_leader","TRB_leader","Linker","Link_order",
               "TRA_5_prime_seq","TRA_3_prime_seq","TRB_5_prime_seq","TRB_3_prime_seq"]]
    return df_reordered

def run_stitchr(expanded_df, output_file):
    input_file = "stitchr_input.tsv"
    expanded_df.to_csv(input_file, index=False, sep="\t")
    command = f"thimble -in {input_file} -o {output_file}"
    subprocess.run(command, shell=True, check=True)
    output_file = output_file + ".tsv"
    tcr_df = pd.read_csv(output_file, sep="\t")
    cols_to_keep = ["TCR_name", "TRA_aa", "TRB_aa", "TRAV", "TRAJ", "TRA_CDR3", "TRBV", "TRBJ", "TRB_CDR3"]
    tcr_df = tcr_df[cols_to_keep]
    tcr_df = tcr_df.rename(columns={"TCR_name": "tcr_id", "TRA_aa": "TCRA", "TRB_aa": "TCRB", "TRA_CDR3": "CDR3a", "TRB_CDR3": "CDR3b"})
    return tcr_df

def fasta_to_df(fasta_file):
    with open(fasta_file) as f:
        sequences = []
        names = []
        sequence = ""  
        for line in f:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""  
                names.append(line.strip()[1:].split(' ')[1])
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return pd.DataFrame({'mhc_allele': names, 'mhc_seq': sequences})

def add_mhc_sequences(tcr_df, hla_col="HLA", seq_col="MHC_seq", mhc_seq_col="mhc_seq"):
    fasta_file = "./data/hla_prot.fasta"
    df_hla = fasta_to_df(fasta_file)
    tcr_df[seq_col] = ""
    for idx, row in tcr_df.iterrows():
        if pd.notna(row[seq_col]) and row[seq_col].strip() != "":
            continue
        base = row[hla_col].strip()
        variants = [base, f"{base}:01", f"{base}:01:01"]
        found = None
        for v in variants:
            match = df_hla.loc[df_hla['mhc_allele'].str.strip() == v]
            if not match.empty:
                found = match[mhc_seq_col].values[0]
                tcr_df.at[idx, seq_col] = found
                break
        if found is None:
            print(f"No match found for {base} in any resolution.")
    return tcr_df

def sample_seed() -> int:
    """Sample a random seed for an AlphaFold3 job."""
    return random.randint(0, 2**32 - 1)

def clean_sequence(seq: str) -> str:
    """Remove '*' characters from a protein sequence."""
    if isinstance(seq, str):
        return seq.replace("*", "")
    return seq

def generate_af3_jsons(
    tcr_df,
    b2m="IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM",
    json_folder="./json_files",
    files_per_folder=20,
    non_pep_msa=False,):
    """
    Generate AlphaFold3 JSON input files for TCR–pMHC complexes.
    Removes '*' characters from all sequences.
    """
    os.makedirs(json_folder, exist_ok=True)
    for idx, (_, row) in enumerate(tcr_df.iterrows()):
        tcr_name = f"TCR_{row['tcr_id']}"
        seed = sample_seed()
        mhc_seq = clean_sequence(row["MHC_seq"])
        epitope_seq = clean_sequence(row["Epitope"])
        tcra_seq = clean_sequence(row["TCRA"])
        tcrb_seq = clean_sequence(row["TCRB"])
        b2m_seq = clean_sequence(b2m)

        if non_pep_msa:
            entry = {
                "name": tcr_name,
                "modelSeeds": [seed],
                "sequences": [
                    {"protein": {"id": "A", "sequence": mhc_seq}},
                    {"protein": {"id": "B", "sequence": b2m_seq}},
                    {
                        "protein": {
                            "id": "C",
                            "sequence": epitope_seq,
                            "pairedMsa": "",
                            "unpairedMsa": "",
                        }
                    },
                    {"protein": {"id": "D", "sequence": tcra_seq}},
                    {"protein": {"id": "E", "sequence": tcrb_seq}},
                ],
                "dialect": "alphafold3",
                "version": 2,
            }
        else:
            entry = {
                "name": tcr_name,
                "modelSeeds": [seed],
                "sequences": [
                    {"protein": {"id": "A", "sequence": mhc_seq}},
                    {"protein": {"id": "B", "sequence": b2m_seq}},
                    {"protein": {"id": "C", "sequence": epitope_seq}},
                    {"protein": {"id": "D", "sequence": tcra_seq}},
                    {"protein": {"id": "E", "sequence": tcrb_seq}},
                ],
                "dialect": "alphafold3",
                "version": 2,}

        batch_index = idx // files_per_folder + 1
        batch_folder = os.path.join(json_folder, f"batch_{batch_index}")
        os.makedirs(batch_folder, exist_ok=True)
        tcr_folder = os.path.join(batch_folder, tcr_name)
        os.makedirs(tcr_folder, exist_ok=True)
        file_path = os.path.join(tcr_folder, f"{tcr_name}.json")
        with open(file_path, "w") as f:
            json.dump(entry, f, indent=4)

    print("✅ AlphaFold3 JSON files successfully generated.")

def main():
    parser = argparse.ArgumentParser(description="Generate AlphaFold3 JSON input files for TCR–pMHC complexes.")
    parser.add_argument("--in_df", required=True, help="Path to the input CSV file containing TCR–pMHC data.")
    parser.add_argument("--out_df", required=True, help="Path to the output CSV file where the processed TCR dataframe will be saved.")
    parser.add_argument("--out", required=True, help="Path to the output folder where JSON files will be saved.")
    parser.add_argument("--nfiles", type=int, default=20, help="Number of JSON files per subfolder (default: 20).")
    parser.add_argument("--run", action="store_true", help="If set, will run the generated AlphaFold3 jobs using SLURM.")
    args = parser.parse_args()

    input_csv = args.in_df
    output_folder = args.out
    files_per_folder = args.nfiles 

    # Read the input file
    tcr_df = pd.read_csv(input_csv)
    compulsory = ["TRAV", "TRAJ", "TRBV", "TRBJ", "CDR3a", "CDR3b", "MHC_allele", "Epitope"]
    if not all(col in tcr_df.columns for col in compulsory):
        missing = [col for col in compulsory if col not in tcr_df.columns]
        raise ValueError(f"Missing compulsory columns: {', '.join(missing)}")

    # If not tcr_id add it as a number
    if "tcr_id" not in tcr_df.columns:
        tcr_df.insert(0, "tcr_id", range(1, len(tcr_df) + 1))

    # If not TCRA or TCRB, generate them using stitchr
    if not all(col in tcr_df.columns for col in ["TCRA", "TCRB"]):
        print("TCRA or TCRB not found, generating them using stitchr...")
        tcr_df_stitchr = reorder_add_columns(tcr_df)
        stitchr_output_file = "stitchr_output"
        tcr_df_stitched = run_stitchr(tcr_df_stitchr, stitchr_output_file)
        tcr_df = pd.merge(tcr_df, tcr_df_stitched, on="tcr_id", how="left")
        tcr_df = tcr_df[[col for col in tcr_df.columns if not col.endswith('_y')]]
        tcr_df = tcr_df.rename(columns={col: col[:-2] for col in tcr_df.columns if col.endswith('_x')})
    
    # Check if MHC_seq is present, if not, add it using fasta_to_df
    if "MHC_seq" not in tcr_df.columns:
        print("MHC_seq not found, adding it using fasta_to_df...")
        tcr_df = add_mhc_sequences(tcr_df, hla_col="MHC_allele", seq_col="MHC_seq", mhc_seq_col="mhc_seq")
    
    # Save tcr_df to csv
    print(f"Saving TCR dataframe with MHC sequences to {args.out_df}...")
    tcr_df.to_csv(args.out_df, index=False)

    # Generate AlphaFold3 JSON files
    print(f"Generating AlphaFold3 JSON files in {output_folder}...")
    generate_af3_jsons(tcr_df, json_folder=output_folder, files_per_folder=files_per_folder)
    
    for file in ["stitchr_input.tsv", "stitchr_output.tsv"]:
        if os.path.exists(file):
            os.remove(file)
    if args.run:
        output_folder = os.path.abspath(output_folder)
        os.makedirs(os.path.join(output_folder, "output"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "logs"), exist_ok=True)
        os.chdir(output_folder)

        subprocess.run("cp ../src/af3_run_mnv.sh .",shell=True,check=True)

        command = f"""
cd {output_folder}
for dir in ./batch_*/; do [ -d "$dir" ] && sbatch -A bsc72 -q acc_bscls af3_run_mnv.sh "$dir" ./output/; done
"""
        print(command)
        #subprocess.run(command, shell=True, executable="/bin/bash")

if __name__ == "__main__":
    main()
