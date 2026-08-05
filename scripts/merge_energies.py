#!/usr/bin/env python3
import argparse, os, pandas as pd, shutil, subprocess
from multiprocessing import Pool
from tqdm import tqdm

def process_tcr(subpath):
    try:
        ene=[e.path for e in os.scandir(subpath) if e.name.endswith(".ene")][0]
        with open(ene) as f:
            for line in f:
                p=line.split()
                if len(p)>=6 and p[0].replace(".","").isdigit():
                    return [os.path.basename(subpath),float(p[1]),float(p[2]),float(p[3]),float(p[4]),float(p[5])]
    except:
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-tcoarse","--input_tcoarse",required=True)
    ap.add_argument("-metadata","--input_metadata",required=False)
    ap.add_argument("-tar","--input_pydock",required=True)
    ap.add_argument("-metrics","--input_metrics",required=False)
    ap.add_argument("-o","--output",required=True)
    a=ap.parse_args()

    base=os.path.splitext(a.input_pydock)[0]
    if os.path.exists(base): shutil.rmtree(base)
    os.makedirs(base)

    subprocess.run(f"tar -xf {a.input_pydock} -C {base}",shell=True,check=True)

    dirs=[os.path.join(base,x) for x in os.listdir(base) if os.path.isdir(os.path.join(base,x))]

    with Pool(8) as p:
        rows=[x for x in tqdm(p.imap_unordered(process_tcr,dirs),total=len(dirs),desc="PyDock") if x]

    df=pd.DataFrame(rows,columns=["tcr_id","Conf","Ele","Desolv","VDW","Total"])
    df["model_number"]=df["tcr_id"].str.rsplit("_",n=1).str[-1].astype(int)
    df["total_w1"]=df["Ele"]+df["Desolv"]+df["VDW"]
    df["tcr_id"]=df["tcr_id"].str.rsplit("_",n=1).str[0] #tcr_17_0 → tcr_17

    df1=pd.read_csv(a.input_tcoarse)
    df1["model_number"]=df1["model_number"].astype(int)

    merged=pd.merge(df1,df,on=["tcr_id","model_number"],how="inner")
    print(merged.columns.tolist())

    if a.input_metadata:
        input_metadata=pd.read_csv(a.input_metadata)
        print(input_metadata.columns.tolist())
        merged=pd.merge(merged,input_metadata,on=["tcr_id", "model_number"],how="left")

    if a.input_metrics:
        input_metrics=pd.read_csv(a.input_metrics)
        print(input_metrics.columns.tolist())
        merged=pd.merge(merged,input_metrics,on=["tcr_id", "model_number"],how="left")

    merged = merged.loc[:, ~merged.columns.str.endswith('_y')]
    merged = merged.rename(columns=lambda x: x[:-2] if x.endswith('_x') else x)

    merged.to_csv(a.output,index=False)

    print(f"PyDock rows: {len(df)}")
    print(f"TCoarse rows: {len(df1)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Saved: {a.output}")

if __name__=="__main__":
    main()
