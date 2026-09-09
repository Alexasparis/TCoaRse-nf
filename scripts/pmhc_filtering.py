#!/usr/bin/env python3
#python pmhc_filtering.py --out_f ../out_f --tcrpmhc ../out_f/tcrpmhc_full.csv

import os
import pandas as pd
import re
import argparse

thresholds = {
    "NetCleave": {"strong": 0.6},
    "mhcflurry_pres": {"strong": 1.0, "weak": 2.0},
    "mhcflurry_proc": {"strong": 1.0, "weak": 2.0},
    "NOAH": {"strong": -5, "weak": -1},
    "PredIG": {"strong": 0.19},
    "EL_rank": {"strong": 0.5, "weak": 2.0},
    "mhcflurry_aff": {"strong": 1.0, "weak": 2.0},
    "BA_rank": {"strong": 0.5, "weak": 2.0}}

lower_better = {
    "NOAH", "EL_rank", "mhcflurry_aff",
    "mhcflurry_pres", "BA_rank", "mhcflurry_proc"}

def tidy_netmhcpan(path):
    rows = []
    allele = None
    data_started = False

    with open(path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("# Allele:"):
                allele = fix_allele_star(line.split(":", 1)[1].strip())
                data_started = False
                continue

            if line.startswith("---"):
                data_started = True
                continue

            if not data_started or allele is None:
                continue

            if line.startswith("Pos"):
                continue

            fields = line.split()

            # Solo aceptar filas reales de predicción
            if len(fields) < 16:
                continue

            if not fields[0].isdigit():
                continue

            try:
                el_score = float(fields[11])
                el_rank = float(fields[12])
                ba_score = float(fields[13])
                ba_rank = float(fields[14])
            except ValueError:
                continue

            rows.append({"peptide": fields[2],"allele": allele,"core": fields[3],"icore": fields[9],
                         "EL_score": el_score,"EL_rank": el_rank,"BA_score": ba_score,"BA_rank": ba_rank,})

    result = pd.DataFrame(rows)
    return result

def fix_allele_star(allele):
    allele = str(allele).strip()
    # HLA-X*NN:NN
    if re.match(r"HLA-[A-Z]\*\d{2}:\d{2}", allele):
        return allele
    # HLA-XNN:NN or HLA-XNNNN (sin *)
    m = re.match(r"HLA-([A-Z])(\d{2}):?(\d{2})?", allele)
    if m:
        gene, g1, g2 = m.groups()
        if g2 is None:
            g2 = '01'  
        return f"HLA-{gene}*{g1}:{g2}"
    # has X*NN:NN no HLA-, add HLA-
    m2 = re.match(r"([A-Z])\*(\d{2}):(\d{2})", allele)
    if m2:
        gene, g1, g2 = m2.groups()
        return f"HLA-{gene}*{g1}:{g2}"
    # has XNNNN or XNN:NN not HLA- nor *, add all
    m3 = re.match(r"([A-Z])(\d{2}):?(\d{2})?", allele)
    if m3:
        gene, g1, g2 = m3.groups()
        if g2 is None:
            g2 = '01'
        return f"HLA-{gene}*{g1}:{g2}"
    return allele

def load_files (netmhc_pan, mhc_flurry, predig):
    netmhcpan = tidy_netmhcpan(netmhc_pan)
    mhcflurry = pd.read_csv(mhc_flurry)
    predig = pd.read_csv(predig)
    
    #print("PredIG shape:", predig.shape)
    #print("NetMHCpan shape:", netmhcpan.shape)
    #print("MHCflurry shape:", mhcflurry.shape)

    netmhcpan["allele"] = netmhcpan["allele"].apply(fix_allele_star)
    mhcflurry["allele"] = mhcflurry["allele"].apply(fix_allele_star)
    predig["allele"] = predig["allele"].apply(fix_allele_star)

    merged = pd.merge(netmhcpan, mhcflurry, on=['peptide', 'allele'], how='inner')
    merged_predig = pd.merge(merged, predig, on=['peptide', 'allele'], how='inner')

    colstokeep = ["peptide", "allele", "EL_score", "EL_rank", "BA_score", "BA_rank", "PredIG", "NOAH", "NetCleave", "mhcflurry_affinity", "mhcflurry_affinity_percentile", "mhcflurry_processing_score", "mhcflurry_presentation_score", "mhcflurry_presentation_percentile"]
    merged_predig = merged_predig[colstokeep]

    # Add a mhcflurry_presentation_score_percentile column with the percentile of the mhcflurry_presentation_score
    merged_predig = merged_predig.sort_values(by="mhcflurry_processing_score", ascending=False)
    merged_predig["mhcflurry_processing_percentile"] = ((1 - merged_predig["mhcflurry_processing_score"].rank(pct=True)) * 100)

    # If Nans add -10 in NOAH, 1 in NetCleave, 1 Predig, 0 in precentiles and rank columns
    merged_predig["EL_rank"] = merged_predig["EL_rank"].fillna(0)
    merged_predig["BA_rank"] = merged_predig["BA_rank"].fillna(0)
    merged_predig["mhcflurry_affinity_percentile"] = merged_predig["mhcflurry_affinity_percentile"].fillna(0)
    merged_predig["mhcflurry_processing_percentile"] = merged_predig["mhcflurry_processing_score"].fillna(0)  
    merged_predig["mhcflurry_presentation_percentile"] = merged_predig["mhcflurry_presentation_percentile"].fillna(0)
    merged_predig["NOAH"] = merged_predig["NOAH"].fillna(-10)
    merged_predig["NetCleave"] = merged_predig["NetCleave"].fillna(1)
    merged_predig["PredIG"] = merged_predig["PredIG"].fillna(1)

    return merged_predig

def apply_filter(df, col, strength):
    threshold = thresholds[col][strength]
    if col in lower_better:
        return df[df[col] <= threshold].copy()
    else:
        return df[df[col] >= threshold].copy()

def apply_filter_preset(df, filter_preset):
    result = df.copy()
    filter_preset = filter_preset[0]
    filters = [f.strip() for f in filter_preset.split("+")]

    for f in filters:
        col, strength = f.rsplit("_", 1)
        result = apply_filter(result,col,strength)
        print(f"{f}: {len(result)} rows remaining")
    return result

def main():
    parser = argparse.ArgumentParser(description="Process netMHCpan, MHCflurry, and PredIG output files.")
    parser.add_argument("--out_f", required=True, help="Path to the output folder where filtered TCR-pMHC data will be saved.")
    parser.add_argument("--tcrpmhcs", required=True, help="Path to the TCR-pMHC input CSV file.")
    args = parser.parse_args()

    # Print thresholds 
    print("Thresholds:")
    for col, thresh in thresholds.items():
        print(f"{col}: {thresh}")

    # allow the user to select a filter or to write a filter in this format: 
    filter_VHIO = "EL_rank_strong"
    filter_relaxed = "BA_rank_strong + NOAH_weak + mhcflurry_pres_strong"
    filter_stringent = "NetCleave_strong + BA_rank_strong + NOAH_weak"

    print("\nAvailable filter presets:")
    print(f"1. VHIO filter: {filter_VHIO}")
    print(f"2. Relaxed filter: {filter_relaxed}")
    print(f"3. Stringent filter: {filter_stringent}")
    print("4. Custom filter")

    choice = input("\nSelect a filter [1-4]: ").strip()

    if choice == "1":
        selected_filter = filter_VHIO
    elif choice == "2":
        selected_filter = filter_relaxed
    elif choice == "3":
        selected_filter = filter_stringent
    elif choice == "4":
        selected_filter = input(
            "Enter custom filter (e.g. 'EL_rank_strong + NOAH_weak'): "
        ).strip()
    else:
        raise ValueError("Invalid filter selection. Choose 1, 2, 3, or 4.")

    print(f"\nSelected filter: {selected_filter}\n")

    tcrpmhcs_df = pd.read_csv(args.tcrpmhcs)
    tcrpmhcs_df['pmhc'] = tcrpmhcs_df['Epitope'] + "_" + tcrpmhcs_df['MHC_allele']
    input_pmhcs = tcrpmhcs_df['pmhc'].unique()

    # Load the output files from the previous steps
    # First see if exists, if not, raise an error
    netmhcpan_path = os.path.join(args.out_f, "netmhcpan_output.txt")
    mhcflurry_path = os.path.join(args.out_f, "mhcflurry_output.csv")
    predig_path = os.path.join(args.out_f, "predig_output.csv")
    if not os.path.exists(netmhcpan_path):
        raise FileNotFoundError(f"File not found: {netmhcpan_path}")
    if not os.path.exists(mhcflurry_path):
        raise FileNotFoundError(f"File not found: {mhcflurry_path}")
    if not os.path.exists(predig_path):
        raise FileNotFoundError(f"File not found: {predig_path}")
    
    merged_df = load_files(netmhcpan_path, mhcflurry_path, predig_path)
    # Remove predig_input.csv, predig_output.csv, mhcflurry_input.csv, mhcflurry_output.csv, alleles_list.txt, netmhcpan_output.txt, peptides.txt
    for file in [os.path.join(args.out_f, "predig_input.csv"), os.path.join(args.out_f, "predig_output.csv"),
                 os.path.join(args.out_f, "mhcflurry_input.csv"), os.path.join(args.out_f, "mhcflurry_output.csv"), 
                 os.path.join(args.out_f, "alleles_list.txt"), os.path.join(args.out_f, "netmhcpan_output.txt"), 
                 os.path.join(args.out_f, "peptides.txt")]:
        if os.path.exists(file):
            os.remove(file)

    merged_df['pmhc'] = merged_df['peptide'] + "_" + merged_df['allele']
    merged_df['pmhc'] = merged_df['pmhc'].str.replace("HLA-", "", regex=False)

    merged_df = merged_df[merged_df['pmhc'].isin(input_pmhcs)].copy()
    merged_df.drop(columns=['pmhc'], inplace=True)
    merged_df = merged_df.drop_duplicates(subset=['peptide', 'allele'])

    print(f"Input TCR-pMHC data contains, {len(merged_df)} unique pMHC pairs.")
    merged_df.to_csv(os.path.join(args.out_f, "pmhc_scores.csv"), index=False)

    # Apply selected filter
    filtered_df = apply_filter_preset(merged_df, [selected_filter])
    filtered_df['pmhc'] = filtered_df['peptide'] + "_" + filtered_df['allele']
    filtered_df['pmhc'] = filtered_df['pmhc'].str.replace("HLA-", "", regex=False)

    print(f"\nTCR-pMHC complexes before filtering pMHCs: {len(tcrpmhcs_df)} rows")
    filtered_tcrpmhcs_df = tcrpmhcs_df[tcrpmhcs_df['pmhc'].isin(filtered_df['pmhc'])].copy()
    filtered_tcrpmhcs_df.drop(columns=['pmhc'], inplace=True)
    print(f"TCR-pMHC complexes after filtering pMHCs: {len(filtered_tcrpmhcs_df)} rows")
    filtered_tcrpmhcs_df.to_csv(os.path.join(args.out_f, "tcrpmhc_filtered.csv"), index=False)
    print(f"✅ Filtered TCR-pMHC data saved to {os.path.join(args.out_f, 'tcrpmhc_filtered.csv')}")
    print("Now run:\n","   python af3_jsons.py --in_df " + os.path.join(args.out_f, "tcrpmhc_filtered.csv") + " --out " + os.path.join(args.out_f, 'af3_jsons'))
    
if __name__ == "__main__":
    main()
    
