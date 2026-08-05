#!/usr/bin/env python3

import os, glob, argparse
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from Bio.PDB import PDBParser, PPBuilder
from anarci import anarci

ppb, parser = PPBuilder(), PDBParser(QUIET=True)

def seq(structure, chain):
    try:
        return "".join(str(p.get_sequence()) for p in ppb.build_peptides(structure[0][chain])) or None
    except:
        return None

def read_pdb(pdb):
    name = os.path.splitext(os.path.basename(pdb))[0]
    s = parser.get_structure("", pdb)
    tcr_id = name.rsplit("_", 1)[0]  # tcr_17_0 → tcr_17
    model_number = int(name.split("_")[-1])  # tcr_17_0 → 0
    return {
        "tcr_id": tcr_id,
        "model_number": model_number,
        "MHC_seq": seq(s, "A"),
        "B2M_seq": seq(s, "B"),
        "Epitope": seq(s, "C"),
        "TCRA": seq(s, "D"),
        "TCRB": seq(s, "E"),}

def cdrs(num):
    c1, c2, c3 = [], [], []
    for p, a in num:
        if a == "-":
            continue
        n = p[0]
        if 26 <= n <= 39:
            c1.append(a)
        elif 55 <= n <= 66:
            c2.append(a)
        elif 104 <= n <= 118:
            c3.append(a)
    return "".join(c1), "".join(c2), "".join(c3)

def annotate(seq):
    if not seq:
        return seq, None, None, None
    try:
        n, _, _ = anarci([("x", seq)], scheme="imgt")
        return (seq, *cdrs(n[0][0][0])) if n[0] else (seq, None, None, None)
    except:
        return seq, None, None, None

def germlines(seq):
    if not seq:
        return "NA", "NA"
    try:
        r = anarci([("x", seq)], scheme="aho", assign_germline=True, output=False)
        g = r[1][0][0]["germlines"]
        return g["v_gene"][0][1], g["j_gene"][0][1]
    except:
        return "NA", "NA"

def fasta_to_df(fasta_file):
    names, seqs, seq = [], [], ""
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                if seq:
                    seqs.append(seq)
                    seq = ""
                names.append(line.strip()[1:].split()[1])
            else:
                seq += line.strip()
        if seq:
            seqs.append(seq)
    return pd.DataFrame({"mhc_allele": names, "MHC_seq": seqs})

def assign_hla(seq, hla_df):
    m = hla_df.loc[hla_df.MHC_seq == seq]
    if not m.empty:
        return m.iloc[0].mhc_allele
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_folder")
    ap.add_argument("-o", "--output", default="tcr_info.csv")
    args = ap.parse_args()

    pdbs = sorted(glob.glob(os.path.join(args.pdb_folder, "*.pdb")))

    df = pd.DataFrame(Parallel(n_jobs=-1)(delayed(read_pdb)(p) for p in tqdm(pdbs, desc="Reading PDBs")))

    alpha = df.TCRA.dropna().unique()
    beta = df.TCRB.dropna().unique()

    alpha_cdr = dict((x[0], x[1:]) for x in Parallel(n_jobs=-1)( delayed(annotate)(s) for s in tqdm(alpha, desc="Alpha CDRs")))
    beta_cdr = dict((x[0], x[1:]) for x in Parallel(n_jobs=-1)(delayed(annotate)(s) for s in tqdm(beta, desc="Beta CDRs")))

    alpha_gl = dict(Parallel(n_jobs=-1)(delayed(lambda s: (s, germlines(s)))(s) for s in tqdm(alpha, desc="Alpha germlines")))
    beta_gl = dict(Parallel(n_jobs=-1)(delayed(lambda s: (s, germlines(s)))(s) for s in tqdm(beta, desc="Beta germlines")))

    df[["CDR1a", "CDR2a", "CDR3a"]] = df.TCRA.map(alpha_cdr).apply(pd.Series)
    df[["CDR1b", "CDR2b", "CDR3b"]] = df.TCRB.map(beta_cdr).apply(pd.Series)

    df[["TRAV", "TRAJ"]] = df.TCRA.map(alpha_gl).apply(pd.Series)
    df[["TRBV", "TRBJ"]] = df.TCRB.map(beta_gl).apply(pd.Series)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data")
    hla_df = fasta_to_df(os.path.join(data_dir, "hla_prot.fasta"))

    hla_df["MHC_allele"] = hla_df["mhc_allele"].str.extract(r"^([^:]+:[^:]+)")
    hla_df = hla_df.drop_duplicates(subset="MHC_seq", keep="first")

    hla_map = dict(zip(hla_df["MHC_seq"], hla_df["MHC_allele"]))
    df["MHC_allele"] = df["MHC_seq"].map(hla_map)
    
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()