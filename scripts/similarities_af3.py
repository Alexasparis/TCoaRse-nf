#!/usr/bin/env python3
import argparse, os, subprocess, tempfile
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Levenshtein import distance as lev_distance
from tcrdist.repertoire import TCRrep
from joblib import Parallel, delayed
from tqdm import tqdm
import uuid
from DockQ.DockQ import load_PDB,run_on_all_native_interfaces
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import islice
import os
import shutil

aligner = PairwiseAligner()
aligner.mode = "local"
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.open_gap_score = -5
aligner.extend_gap_score = -0.5

def parse_general_file(general_file):
    df = pd.read_csv(general_file, sep='\t')
    pdb_dict = {}

    for pdb_id, group in df.groupby('pdb.id'):
        pdb_id = pdb_id.split('.')[0]
        chains = {
            'tcra_chain': None,
            'tcrb_chain': None,
            'peptide_chain': None,
            'mhc_chain': None,
            'b2_chain': None}
        for _, row in group.iterrows():
            chain_id = row['chain.id']
            chain_type = row['chain.type']
            chain_component = row['chain.component']
            chain_supertype = row['chain.supertype']

            if chain_component == 'TCR' and chain_type == 'TRA':
                chains['tcra_chain'] = chain_id
            elif chain_component == 'TCR' and chain_type == 'TRB':
                chains['tcrb_chain'] = chain_id
            elif chain_component == 'PEPTIDE':
                chains['peptide_chain'] = chain_id
            elif chain_component == 'MHC' and chain_supertype == 'MHCI' and chain_type == 'MHCa':
                chains['mhc_chain'] = chain_id
            elif chain_component == 'MHC' and chain_supertype == 'MHCI' and chain_type == 'MHCb':
                chains['b2_chain'] = chain_id

        pdb_dict[pdb_id] = chains
    return pdb_dict

def sw(s1, s2):
    if pd.isna(s1) or pd.isna(s2): return np.nan
    return aligner.score(s1, s2) / (2 * min(len(s1), len(s2)))

def lev(s1, s2):
    if pd.isna(s1) or pd.isna(s2): return np.nan
    return 1.0 if max(len(s1), len(s2)) == 0 else 1 - lev_distance(s1, s2) / max(len(s1), len(s2))

def stats(x, p):
    x = np.asarray(x, float)
    return {f"{p}_min": np.nanmin(x), f"{p}_mean": np.nanmean(x), f"{p}_max": np.nanmax(x)}

def remove_headers(file_path):
    with open(file_path) as f:
        return [l for l in f if l.startswith("ATOM") and len(l) > 21]

def get_chains(lines):
    return set([x[21].strip() if x[21].strip() else "?" for x in lines])

def merge_pdb(pdb_file,chain_mapping):
    base=os.path.splitext(os.path.basename(pdb_file))[0]
    tmpdir=tempfile.mkdtemp(prefix=f"{uuid.uuid4().hex[:8]}_")
    cleaned=os.path.join(tmpdir,f"{base}.pdb")
    receptor=os.path.join(tmpdir,"A.pdb")
    ligand=os.path.join(tmpdir,"B.pdb")
    merged=os.path.join(tmpdir,f"{base}_merged.pdb")
    lines=remove_headers(pdb_file)
    with open(cleaned,"w") as f:f.writelines(lines)
    present=get_chains(lines)
    required=[chain_mapping["mhc_chain"],chain_mapping["peptide_chain"],chain_mapping["tcra_chain"],chain_mapping["tcrb_chain"]]
    missing=set([x for x in required if x not in [None,"?"]])-present
    if missing:raise ValueError(f"{base}: faltan cadenas {missing}, presentes={present}, mapping={chain_mapping}")
    receptor_chains=",".join([x for x in [chain_mapping["mhc_chain"],chain_mapping["b2_chain"],chain_mapping["peptide_chain"]] if x not in [None,"?"]])
    ligand_chains=",".join([x for x in [chain_mapping["tcra_chain"],chain_mapping["tcrb_chain"]] if x not in [None,"?"]])
    subprocess.run(f"pdb_selchain -{receptor_chains} {cleaned} | pdb_chain -A | pdb_reres -1 | pdb_delhetatm > {receptor}",shell=True,check=True)
    subprocess.run(f"pdb_selchain -{ligand_chains} {cleaned} | pdb_chain -B | pdb_reres -1 | pdb_delhetatm > {ligand}",shell=True,check=True)
    receptor_lines=remove_headers(receptor)
    ligand_lines=remove_headers(ligand)
    if not receptor_lines or not ligand_lines:raise ValueError(f"{base}: receptor o ligand vacio")
    with open(merged,"w") as f:f.writelines(receptor_lines+ligand_lines)
    return merged,tmpdir

def prepare_merged(pdb_dir,merged_dir,chain_dict,pdb_filter=None):
    os.makedirs(merged_dir,exist_ok=True)
    default={"tcra_chain":"D","tcrb_chain":"E","mhc_chain":"A","b2_chain":"B","peptide_chain":"C"}
    pdbs=[x for x in os.listdir(pdb_dir) if x.endswith(".pdb")]
    if pdb_filter is not None:pdbs=[x for x in pdbs if x in pdb_filter]
    for pdb in tqdm(pdbs,desc="Preparing merged PDBs"):
        base=pdb.replace(".pdb","")
        out=os.path.join(merged_dir,f"{base}_merged.pdb")
        if os.path.exists(out):continue
        pdb_id=base.split("_")[0] if "_" not in base else base.rsplit("_",1)[0]
        mapping=chain_dict.get(pdb_id,default)
        try:
            merged,_=merge_pdb(os.path.join(pdb_dir,pdb),mapping)
            shutil.copy(merged,out)
        except Exception as e:
            print(f"Skipping {pdb}: {e}")

CHAIN_MAP={"A":"A","B":"B"}

def score_pair(post_id,pdb_id,merged_post,merged_pdbs):
    try:
        model_path=os.path.join(merged_post,f"{post_id}_merged.pdb")
        native_path=os.path.join(merged_pdbs,f"{pdb_id}_merged.pdb")
        model=load_PDB(model_path)
        native=load_PDB(native_path)
        info,total=run_on_all_native_interfaces(model,native,chain_map=CHAIN_MAP)
        r=next(iter(info.values()))
        return [post_id,pdb_id,r.get("DockQ",total),r.get("iRMSD"),r.get("LRMSD"),r.get("fnat"),r.get("clashes")]
    except Exception as e:
        print(f"FAILED inside DockQ: {post_id} vs {pdb_id}: {e}",flush=True)
        return [post_id,pdb_id,None,None,None,None,None]

def score_chunk(chunk):
    return [score_pair(*x) for x in chunk]

def chunked(it,size):
    while True:
        x=list(islice(it,size))
        if not x:return
        yield x

def build_jobs(merged_post,merged_pdbs):
    post=[x.replace("_merged.pdb","") for x in os.listdir(merged_post) if x.endswith("_merged.pdb")]
    ref=[x.replace("_merged.pdb","") for x in os.listdir(merged_pdbs) if x.endswith("_merged.pdb")]
    return [(p,r,merged_post,merged_pdbs) for r in ref for p in post]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-pre",required=True);ap.add_argument("-post",required=True);ap.add_argument("-pre_pdb",required=True);ap.add_argument("-post_pdb",required=True);ap.add_argument("-o_seq",required=True);ap.add_argument("-o_str",required=True);ap.add_argument("-n_jobs",type=int,default=-1)
    args=ap.parse_args()
    current_dir=os.path.dirname(os.path.abspath(__file__))
    chain_dict=parse_general_file(f"{current_dir}/../data/chain_info.txt")
    af3_mapping={"tcra_chain":"D","tcrb_chain":"E","mhc_chain":"A","b2_chain":"B","peptide_chain":"C"}
    cols=["CDR3a","CDR3b","TRAV","TRAJ","TRBV","TRBJ","MHC_seq","Epitope"]
    pre=pd.read_csv(args.pre);post=pd.read_csv(args.post)
    pre_seq=pre.drop_duplicates("tcr_id").reset_index(drop=True);post_seq=post.drop_duplicates("tcr_id").reset_index(drop=True)
    cell_df=pd.concat([post_seq[cols],pre_seq[cols]],ignore_index=True).rename(columns={"CDR3a":"cdr3_a_aa","CDR3b":"cdr3_b_aa","TRAV":"v_a_gene","TRAJ":"j_a_gene","TRBV":"v_b_gene","TRBJ":"j_b_gene"})
    cell_df["count"]=1
    tr=TCRrep(cell_df=cell_df,organism="human",chains=["alpha","beta"],compute_distances=True)
    n=len(post_seq)
    pre_trav=set(pre_seq.TRAV.dropna());pre_traj=set(pre_seq.TRAJ.dropna());pre_trbv=set(pre_seq.TRBV.dropna());pre_trbj=set(pre_seq.TRBJ.dropna());pre_epi=set(pre_seq.Epitope.dropna())
    def seq_process(i):
        row=post_seq.iloc[i];f={"tcr_id":row.tcr_id}
        f.update(stats(tr.pw_cdr3_a_aa[i,n:],"tcrdist_cdr3a"));f.update(stats(tr.pw_cdr3_b_aa[i,n:],"tcrdist_cdr3b"));f.update(stats([sw(row.MHC_seq,x) for x in pre_seq.MHC_seq],"sim_MHC"));f.update(stats([lev(row.Epitope,x) for x in pre_seq.Epitope],"sim_Epitope"))
        f["TRAV_overlap"]=row.TRAV in pre_trav;f["TRAJ_overlap"]=row.TRAJ in pre_traj;f["TRBV_overlap"]=row.TRBV in pre_trbv;f["TRBJ_overlap"]=row.TRBJ in pre_trbj;f["Epitope_overlap"]=row.Epitope in pre_epi
        return f
    pd.DataFrame(Parallel(n_jobs=args.n_jobs)(delayed(seq_process)(i) for i in tqdm(range(n),desc="Sequence"))).to_csv(args.o_seq,index=False)
    merged_post=os.path.join(os.path.dirname(args.post_pdb),os.path.basename(args.post_pdb)+"_merged")
    merged_pre=os.path.join(os.path.dirname(args.pre_pdb),os.path.basename(args.pre_pdb)+"_merged")
    prepare_merged(args.post_pdb,merged_post,{x.split(".")[0]:af3_mapping for x in os.listdir(args.post_pdb) if x.endswith(".pdb")})
    valid_pdbs=set(pre["tcr_id"].astype(str))
    pre_filtered=[x for x in os.listdir(args.pre_pdb) if x.endswith(".pdb") and os.path.splitext(x)[0] in valid_pdbs]
    prepare_merged(args.pre_pdb,merged_pre,chain_dict,pre_filtered)

    jobs=build_jobs(merged_post,merged_pre)
    print(f"DockQ comparisons: {len(jobs)}",flush=True)
    results=[]

    with ProcessPoolExecutor(max_workers=args.n_jobs if args.n_jobs>0 else 8) as ex:
        futures={ex.submit(score_pair,*x):x for x in jobs}
        results=[]
        for f in tqdm(as_completed(futures),total=len(futures),desc="DockQ"):
            job=futures[f]
            try:
                results.append(f.result())
            except Exception as e:
                print(f"FAILED DockQ: post={job[0]} ref={job[1]} error={e}",flush=True)
            
    dockq_df=pd.DataFrame(results,columns=["tcr_id","ref_id","DockQ","iRMSD","LRMSD","Fnat","clashes"])
    struct_df=dockq_df.groupby("tcr_id").agg({"DockQ":["max","mean"],"iRMSD":["min","mean"],"LRMSD":["min","mean"],"Fnat":["max","mean"],"clashes":["min","mean"]}).reset_index()
    struct_df.columns=["tcr_id","DockQ_max","DockQ_mean","iRMSD_min","iRMSD_mean","LRMSD_min","LRMSD_mean","Fnat_max","Fnat_mean","clashes_min","clashes_mean"]
    struct_df.to_csv(args.o_str,index=False)

if __name__ == "__main__":
    main()
