#!/usr/bin/env python3
#python stitch.py --in_df ../input.csv --out_f ../out_f

import subprocess
import pandas as pd
import argparse
import os
import subprocess

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
    os.remove(input_file)
    os.remove(output_file)
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

def add_mhc_sequences(tcr_df, hla_col="MHC_allele", seq_col="MHC_seq", mhc_seq_col="mhc_seq"):
    df_hla = fasta_to_df("../data/hla_prot.fasta").drop_duplicates("mhc_allele")
    lookup = dict(zip(df_hla["mhc_allele"].str.strip(), df_hla[mhc_seq_col]))
    hla = tcr_df[hla_col].str.strip()
    tcr_df[seq_col] = hla.map(lookup).fillna((hla + ":01").map(lookup)).fillna((hla + ":01:01").map(lookup)).fillna("")
    for x in tcr_df.loc[tcr_df[seq_col].eq(""), hla_col].dropna().unique(): print(f"No match found for {x} in any resolution.")
    return tcr_df

def inputs_pmhc(df, out_folder):

    # NetMHCpan input
    peptides = df["Epitope"].dropna().unique()
    with open(os.path.join(out_folder, "peptides.txt"), "w") as f:
        f.write("\n".join(peptides))

    alleles = df["MHC_allele"].dropna().unique()
    alleles = [f"HLA-{a.replace('*','').replace(' ','')}" for a in alleles]
    with open(os.path.join(out_folder, "alleles_list.txt"), "w") as f:
        f.write(",".join(alleles))

    # MHC flurry input
    pmhc_df = df[["Epitope", "MHC_allele"]].dropna().drop_duplicates()
    # rename as peptide and allele for mhcflurry
    pmhc_df = pmhc_df.rename(columns={"Epitope": "peptide", "MHC_allele": "allele"})
    pmhc_df.to_csv(os.path.join(out_folder, "mhcflurry_input.csv"), index=False, header=True)

    # PredIG input
    script_dir = os.path.dirname(os.path.abspath(__file__))
    human_index = os.path.join(script_dir, "..", "data", "human_index")
    human_index = os.path.abspath(human_index)

    subprocess.run(["java","-jar", os.path.join(script_dir, "PeptideMatchCMD_1.1.jar"),"-a", "query", "-i", human_index,
                    "-Q", os.path.join(out_folder, "peptides.txt"),"-l","-o", os.path.join(out_folder, "peptide_matches.txt")], check=True)
    
    matches_df = pd.read_csv(os.path.join(out_folder, "peptide_matches.txt"),sep="\t", skiprows=2, 
                             header=None,names=["Query", "Subject", "SubjectLength", "MatchStart", "MatchEnd"],usecols=range(5))

    # Remove peptide_matches.txt
    os.remove(os.path.join(out_folder, "peptide_matches.txt"))

    matches_df["is_sp"] = matches_df["Subject"].str.startswith("sp|")
    matches_df = matches_df.sort_values(["Query", "is_sp"], ascending=[True, False]).drop_duplicates("Query")
    matches_df["uniprot_id"] = matches_df["Subject"].str.split("|").str[1]
    matches_df = matches_df.drop(columns="is_sp")
   
    df = df.merge(matches_df[["Query", "uniprot_id"]],left_on="Epitope",right_on="Query",how="left").drop(columns="Query")
    df = df[["Epitope", "MHC_allele", "uniprot_id"]].rename(columns={"Epitope": "epitope", "MHC_allele": "HLA_allele", "uniprot_id": "uniprot_id"})
    df["HLA_allele"] = df["HLA_allele"].apply(lambda x: f"HLA-{x}" if pd.notna(x) else x)

    # Remove last empty line if present
    path = os.path.join(out_folder, "predig_input.csv")
    df.to_csv(path, index=False)
    with open(path, "rb+") as f:
        f.seek(-1, 2)
        if f.read(1) == b"\n": f.truncate(f.tell() - 1)

    # Print how many peptides have not uniprot_id
    no_uniprot = df["uniprot_id"].isna().sum()
    if no_uniprot > 0:
        print(f"Warning: {no_uniprot} peptides do not have a uniprot_id in the PredIG input file.")
    return df

def run_commands(out_folder):
    predig_path = "/Users/alexascunceparis/Desktop/BSC/d_programs/PredIG"
    netmhcpan_path = "/Users/alexascunceparis/Desktop/BSC/d_programs/netMHCpan-4.2/netMHCpan"

    # NetMHCpan:
    hla_list_path = os.path.join(out_folder, "alleles_list.txt")
    with open(hla_list_path, "r") as f:
        hla_list = f.read().strip()
    netmhcpan_output_path = os.path.join(out_folder, "netmhcpan_output.txt")

    netmhcpan_command = f"{netmhcpan_path} -p {os.path.join(out_folder, 'peptides.txt')} -a {hla_list} -BA > {netmhcpan_output_path}"
    print(f"NetMHCpan command: {netmhcpan_command}")

    if not os.path.isfile(netmhcpan_path):
        print(f"Warning: netMHCpan executable not found at {netmhcpan_path}. Please ensure that netMHCpan is installed and available in your PATH.")
    try:
        subprocess.run(netmhcpan_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running netMHCpan: {e}")
        print("Please ensure that netMHCpan is installed and available in your PATH.")
    
    # MHCflurry:
    mhcflurry_input_path = os.path.join(out_folder, "mhcflurry_input.csv")
    mhcflurry_output_path = os.path.join(out_folder, "mhcflurry_output.csv")

    mhcflurry_command = f"mhcflurry-predict {mhcflurry_input_path} --out {mhcflurry_output_path}"
    print(f"\nMHCflurry command: {mhcflurry_command}")

    try:
        subprocess.run(mhcflurry_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running mhcflurry-predict: {e}")
        print("Please ensure that MHCflurry is installed and available in your PATH.")

    # PredIG: 
    predig_input_path = os.path.join(out_folder, "predig_input.csv")
    
    predig_command = f"Rscript {predig_path}/scripts/predig_pipe1_container.R --input {predig_input_path} --out {out_folder} --model neoant --exp_name predig_output"
    print(f"\nPredIG command: {predig_command}")

    if not os.path.isfile(predig_path):
        print(f"Warning: PredIG path not found at {predig_path}. Please ensure that PredIG is installed and available in your PATH.")
    try:
        subprocess.run(predig_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running PredIG: {e}")
        print("Please ensure that R and the required packages are installed and available in your PATH.")

def main():
    parser = argparse.ArgumentParser(description="Generate AlphaFold3 JSON input files for TCR–pMHC complexes.")
    parser.add_argument("--in_df", required=True, help="Path to the input CSV file containing TCR–pMHC data.")
    parser.add_argument("--out_f", required=True, help="Path to the output folder where filtered TCR-pMHC data will be saved.")
    args = parser.parse_args()

    # Parse args
    input_csv = args.in_df
    output_folder = args.out_f
    os.makedirs(output_folder, exist_ok=True)

    # Read the input file
    tcr_df = pd.read_csv(input_csv)

    # TCR: gene information or sequences
    tcr_gene_cols = ["TRAV", "TRAJ", "TRBV", "TRBJ", "CDR3a", "CDR3b"]
    tcr_seq_cols = ["TCRA", "TCRB"]

    if not (all(c in tcr_df.columns for c in tcr_gene_cols) or
            all(c in tcr_df.columns for c in tcr_seq_cols)):
        raise ValueError(f"Input CSV must contain either {tcr_gene_cols} or {tcr_seq_cols}.")

    # MHC: allele or sequence
    if not ("MHC_allele" in tcr_df.columns or "MHC_seq" in tcr_df.columns):
        raise ValueError("Input CSV must contain either 'MHC_allele' or 'MHC_seq'.")

    # Epitope
    if "Epitope" not in tcr_df.columns:
        raise ValueError("Input CSV must contain 'Epitope'.")

    # If not tcr_id add it as a number
    if "tcr_id" not in tcr_df.columns:
        tcr_df.insert(0, "tcr_id", range(1, len(tcr_df) + 1))

    # If not TCRA or TCRB, generate them using stitchr
    if not all(col in tcr_df.columns for col in ["TCRA", "TCRB"]):
        print("TCRA or TCRB not found, generating them using stitchr...")
        tcr_df_stitchr = reorder_add_columns(tcr_df)
        stitchr_input_file = "stitchr_input.tsv"
        tcr_df_stitchr.to_csv(stitchr_input_file, index=False)
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
    print(f"Saving TCR dataframe with MHC sequences to {os.path.join(args.out_f, 'tcrpmhc_full.csv')}...")
    tcr_df.to_csv(os.path.join(args.out_f, "tcrpmhc_full.csv"), index=False)

    # Generate inputs pmhcs
    print(f"Generating inputs for pMHC predictions in {output_folder}...")
    tcr_df = inputs_pmhc(tcr_df, out_folder=output_folder)

    # Run the commands for NetMHCpan, MHCflurry and PredIG
    run_commands(out_folder=output_folder)

    # Print instructions to run pmhc_filtering.py
    print("Now run:\n","python pmhc_filtering.py --out_f " + output_folder + " --tcrpmhc " + os.path.join(output_folder, "tcrpmhc_full.csv"))

if __name__ == "__main__":
    main()