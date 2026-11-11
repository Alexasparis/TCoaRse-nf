#!/usr/bin/env python3
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from Bio.Align import PairwiseAligner
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein
from tcrdist.repertoire import TCRrep
from tqdm import tqdm
from DockQ.DockQ import load_PDB,run_on_all_native_interfaces
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from config import PDB_INFO_CSV, PDB_DIR

CHAIN_MAP={"A":"A","B":"B"}
AF3={"tcra_chain":"D","tcrb_chain":"E","mhc_chain":"A","b2_chain":"B","peptide_chain":"C"}
NANV=np.full(5,np.nan,dtype=np.float32)

def parse_general_file(general_file):
    df=pd.read_csv(general_file,sep="\t")
    out={}
    for pid,cid,comp,typ,sup in zip(df["pdb.id"],df["chain.id"],df["chain.component"],df["chain.type"],df["chain.supertype"]):
        c=out.setdefault(pid.split(".")[0],{"tcra_chain":None,"tcrb_chain":None,"peptide_chain":None,"mhc_chain":None,"b2_chain":None})
        if comp=="TCR" and typ=="TRA": c["tcra_chain"]=cid
        elif comp=="TCR" and typ=="TRB": c["tcrb_chain"]=cid
        elif comp=="PEPTIDE": c["peptide_chain"]=cid
        elif comp=="MHC" and sup=="MHCI" and typ=="MHCa": c["mhc_chain"]=cid
        elif comp=="MHC" and sup=="MHCI" and typ=="MHCb": c["b2_chain"]=cid
    return out

def agg(M,w):
    ok=~np.isnan(M)
    den=ok@w
    bad=den==0
    mn=np.where(ok,M,np.inf).min(1)
    mx=np.where(ok,M,-np.inf).max(1)
    me=np.where(ok,M,0)@w/np.where(bad,np.nan,den)
    mn[bad]=np.nan
    mx[bad]=np.nan
    return mn,me,mx

def calculate_tcrdist(pre,post,nj):
    cols=["tcr_id","CDR3a","CDR3b","TRAV","TRBV","TRAJ","TRBJ","CDR1a","CDR1b","CDR2a","CDR2b"]
    ren={"tcr_id":"clone_id","CDR3a":"cdr3_a_aa","CDR3b":"cdr3_b_aa","TRAV":"v_a_gene","TRBV":"v_b_gene","TRAJ":"j_a_gene","TRBJ":"j_b_gene","CDR1a":"cdr1_a_aa","CDR1b":"cdr1_b_aa","CDR2a":"cdr2_a_aa","CDR2b":"cdr2_b_aa"}
    q=post[cols].rename(columns=ren).copy()
    db=pre[cols].rename(columns=ren).copy()
    q["count"]=1
    db["count"]=1
    q["pmhc_a_aa"]="AAAAA"
    q["pmhc_b_aa"]="AAAAA"
    db["pmhc_a_aa"]="AAAAA"
    db["pmhc_b_aa"]="AAAAA"
    tr=TCRrep(cell_df=db,organism="human",chains=["alpha","beta"],db_file="alphabeta_gammadelta_db.tsv",infer_cdrs=False,infer_all_genes=False,deduplicate=False,compute_distances=False,cpus=nj)
    tr.compute_rect_distances(df=q,df2=db)
    specs = {
        "CDR1a": tr.rw_cdr1_a_aa,
        "CDR2a": tr.rw_cdr2_a_aa,
        "CDR3a": tr.rw_cdr3_a_aa,
        "CDR1b": tr.rw_cdr1_b_aa,
        "CDR2b": tr.rw_cdr2_b_aa,
        "CDR3b": tr.rw_cdr3_b_aa,}

    res = {"tcr_id": post["tcr_id"].to_numpy()}
    for region, matrix in specs.items():
        values = np.asarray(matrix, dtype=np.float32)
        res[f"tcrdist_{region}_min"] = np.nanmin(values, axis=1)
        res[f"tcrdist_{region}_mean"] = np.nanmean(values, axis=1)
        res[f"tcrdist_{region}_max"] = np.nanmax(values, axis=1)

    return res

_REF=None
_AL=None

def init_sw(ref):
    global _REF,_AL
    _REF=ref
    _AL=PairwiseAligner(mode="local",match_score=2,mismatch_score=-1,open_gap_score=-5,extend_gap_score=-0.5)

def sw_rows(qs):
    return np.array([[np.nan if (not q or not r) else _AL.score(q,r)/(2*min(len(q),len(r))) for r in _REF] for q in qs],dtype=np.float32)

def sw_matrix(q,r,nj):
    if not q: return np.empty((0,len(r)),dtype=np.float32)
    ch=[c for c in np.array_split(np.array(q,dtype=object),min(len(q),nj*4)) if len(c)]
    with ProcessPoolExecutor(nj,initializer=init_sw,initargs=(r,)) as ex:
        return np.vstack(list(ex.map(sw_rows,ch)))

def seq_sim(pre,post,col,name,kind,nj):
    pc,pu=pd.factorize(pre[col])
    qc,qu=pd.factorize(post[col])
    w=np.bincount(pc[pc>=0],minlength=len(pu)).astype(np.float32)
    pu=[str(x) for x in pu]
    qu=[str(x) for x in qu]
    M=sw_matrix(qu,pu,nj) if kind=="sw" else cdist(qu,pu,scorer=Levenshtein.normalized_similarity,workers=nj,dtype=np.float32)
    return {f"sim_{name}_{stat}":np.append(v,np.nan)[qc] for stat,v in zip(("min","mean","max"),agg(M,w))}

def renumber(lines,chains,new):
    out=[]
    last=None
    n=0
    for l in lines:
        if l[21] in chains:
            key=l[21:27]
            if key!=last:
                n+=1
                last=key
            out.append(f"{l[:21]}{new}{n:4d} {l[27:]}")
    return out

def merge_pdb(path,mp):

    lines=[l for l in Path(path).read_text().splitlines(True) if l.startswith("ATOM") and len(l)>21]
    present={l[21].strip() or "?" for l in lines}
    rec={x for x in (mp["mhc_chain"],mp["b2_chain"],mp["peptide_chain"]) if x not in (None,"?")}
    lig={x for x in (mp["tcra_chain"],mp["tcrb_chain"]) if x not in (None,"?")}
    missing={x for x in (mp["mhc_chain"],mp["peptide_chain"],mp["tcra_chain"],mp["tcrb_chain"]) if x not in (None,"?")}-present
    if missing: raise ValueError(f"faltan cadenas {missing}, presentes={present}, mapping={mp}")
    rl=renumber(lines,rec,"A")
    ll=renumber(lines,lig,"B")
    if not rl or not ll: raise ValueError("receptor o ligand vacio")
    return rl+ll

def prep_one(job):
    src,out,mp=job
    try:
        Path(out+".tmp").write_text("".join(merge_pdb(src,mp)))
        os.replace(out+".tmp",out)
        return None
    except Exception as e:
        return f"{os.path.basename(src)}: {e}"

def prepare_merged(pdb_dir,merged_dir,chain_dict,flt,ex):
    os.makedirs(merged_dir,exist_ok=True)
    jobs=[(os.path.join(pdb_dir,p),os.path.join(merged_dir,p[:-4]+"_merged.pdb"),chain_dict.get(p[:-4].rsplit("_",1)[0],AF3)) for p in os.listdir(pdb_dir) if p.endswith(".pdb") and (flt is None or p in flt)]
    jobs=[j for j in jobs if not os.path.exists(j[1])]
    for m in tqdm(ex.map(prep_one,jobs,chunksize=16),total=len(jobs),desc=f"Merging {os.path.basename(merged_dir)}"):
        if m: print(f"Skipping {m}",flush=True)

def safe_load(path):
    try:
        return load_PDB(path)
    except Exception as e:
        print(f"FAILED load {path}: {e}",flush=True)
        return None

def score_pair(model,native,p,r):
    if model is None or native is None: return NANV
    try:
        info,total=run_on_all_native_interfaces(model,native,chain_map=CHAIN_MAP)
        x=next(iter(info.values()))
        return np.array([x.get("DockQ",total),x.get("iRMSD"),x.get("LRMSD"),x.get("fnat"),x.get("clashes")],dtype=np.float32)
    except Exception as e:
        print(f"FAILED DockQ: {p} vs {r}: {e}",flush=True)
        return NANV

def score_block(refs,posts,start,mpost,mpre):
    models=[safe_load(os.path.join(mpost,f"{p}_merged.pdb")) for p in posts]
    out=np.full((len(refs),len(posts),5),np.nan,dtype=np.float32)
    for i,r in enumerate(refs):
        native=safe_load(os.path.join(mpre,f"{r}_merged.pdb"))
        for j,(p,m) in enumerate(zip(posts,models)):
            out[i,j]=score_pair(m,native,p,r)
    return start,out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-pre",default=PDB_INFO_CSV,required=False)
    ap.add_argument("-pdb_info",required=True)
    ap.add_argument("-pre_pdb", default=PDB_DIR, required=False)
    ap.add_argument("-pdb_dir",required=True)
    ap.add_argument("-out_seq",required=True)
    ap.add_argument("-out_str",required=True)
    ap.add_argument("-workers",type=int,default=-1)
    ap.add_argument("-block",type=int,default=32)
    a=ap.parse_args()
    nj=a.workers if a.workers>0 else (os.cpu_count() or 8)
    bs=a.block
    here=os.path.dirname(os.path.abspath(__file__))
    chain_dict=parse_general_file(f"{here}/../data/chain_info.txt")
    pre=pd.read_csv(a.pre,dtype={"tcr_id":str})
    post=pd.read_csv(a.pdb_info,dtype={"tcr_id":str})

    pre_seq=pre.drop_duplicates("tcr_id").reset_index(drop=True)
    post_seq=post.drop_duplicates("tcr_id").reset_index(drop=True)

    for name,df in [("pre",pre_seq),("post",post_seq)]:
        print(f"\n{name}")
        for c in ["TCRA","TCRB","CDR1a","CDR2a","CDR3a","CDR1b","CDR2b","CDR3b","TRAV","TRAJ","TRBV","TRBJ"]:
            if c in df:
                print(c, "NaN =", df[c].isna().sum(), "dtype =", df[c].dtype)
    cols=["TCRA","TCRB","CDR1a","CDR2a","CDR3a","CDR1b","CDR2b","CDR3b","TRAV","TRAJ","TRBV","TRBJ"]
    pre_seq=pre_seq.dropna(subset=cols).copy()
    post_seq=post_seq.dropna(subset=cols).copy()

    print("Computing TCRdist similarity...",flush=True)
    seq=calculate_tcrdist(pre_seq,post_seq,nj)

    print("Computing MHC/epitope similarity...",flush=True)
    seq.update(seq_sim(pre_seq,post_seq,"MHC_seq","MHC","sw",nj))
    seq.update(seq_sim(pre_seq,post_seq,"Epitope","Epitope","lev",nj))
    pd.DataFrame(seq).to_csv(a.out_seq,index=False)
    pd.DataFrame(seq).to_csv(a.out_str,index=False)

    mpost = os.path.normpath(a.pdb_dir) + "_merged"
    mpre = os.path.normpath(a.pre_pdb) + "_merged"
    valid = set(pre["tcr_id"].astype(str) + "_" + pre["model_number"].astype(int).astype(str)) if "model_number" in pre.columns else set(pre["tcr_id"].astype(str))

    print("PRE files:", len([x for x in os.listdir(a.pre_pdb) if x.endswith(".pdb")]), flush=True)
    print("POST files:", len([x for x in os.listdir(a.pdb_dir) if x.endswith(".pdb")]), flush=True)
    print("VALID:", len(valid), list(valid)[:10], flush=True)
    with ProcessPoolExecutor(nj) as ex:
        prepare_merged(a.pdb_dir,mpost,{},None,ex)
        #prepare_merged(a.pre_pdb,mpre,chain_dict,{f"{t}.pdb" for t in valid},ex)
        print("MERGED PRE:", os.listdir(mpre)[:10], flush=True)
        print("VALID MATCHES:", [x[:-11] for x in os.listdir(mpre) if x.endswith("_merged.pdb") and x[:-11] in valid][:10], flush=True)
        posts=sorted(x[:-11] for x in os.listdir(mpost) if x.endswith("_merged.pdb"))
        refs=sorted(x[:-11] for x in os.listdir(mpre) if x.endswith("_merged.pdb") and x[:-11] in valid)
        n=len(posts)
        pblocks=[(i,posts[i:i+bs]) for i in range(0,n,bs)]
        jobs=[(refs[j:j+bs],pb,i,mpost,mpre) for j in range(0,len(refs),bs) for i,pb in pblocks]
        if not jobs: raise SystemExit("No hay pares para DockQ")
        print(f"DockQ comparisons: {len(refs)*n} en {len(jobs)} bloques",flush=True)
        mn=np.full((n,5),np.nan,dtype=np.float32)
        mx=mn.copy()
        sm=np.zeros((n,5))
        ct=np.zeros((n,5))
        for s,arr in tqdm(ex.map(score_block,*zip(*jobs)),total=len(jobs),desc="DockQ"):
            sl=slice(s,s+arr.shape[1])
            mn[sl]=np.fmin(mn[sl],np.fmin.reduce(arr,axis=0))
            mx[sl]=np.fmax(mx[sl],np.fmax.reduce(arr,axis=0))
            sm[sl]+=np.nansum(arr,axis=0)
            ct[sl]+=(~np.isnan(arr)).sum(0)
    me=sm/np.where(ct>0,ct,np.nan)
    struct=pd.DataFrame({"tcr_id":posts,"DockQ_max":mx[:,0],"DockQ_mean":me[:,0],"iRMSD_min":mn[:,1],"iRMSD_mean":me[:,1],"LRMSD_min":mn[:,2],"LRMSD_mean":me[:,2],"Fnat_max":mx[:,3],"Fnat_mean":me[:,3],"clashes_min":mn[:,4],"clashes_mean":me[:,4]})
    # Divide tcr_id into tcr_id and model_number by last _
    struct[["tcr_id","model_number"]]=struct["tcr_id"].str.rsplit("_",n=1,expand=True)
    struct.to_csv(a.out_str,index=False)

if __name__=="__main__":
    main()