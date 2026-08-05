#!/usr/bin/env python3
import os
import time
import argparse
from pathlib import Path
import h5py
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from anarci import anarci
from transformers import AutoModelForMaskedLM, AutoTokenizer

cdr_ranges = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (104, 118)} 

def load_model(device):
    model_path = "/gpfs/projects/bsc72/weights/ESMC-6B"
    model_name = "biohub/ESMC-6B"

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
        dtype = torch.bfloat16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    source = model_path if os.path.isdir(model_path) else model_name
    print(f"Loading {source} (dtype={dtype}, device={device})")

    model = AutoModelForMaskedLM.from_pretrained(source,dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        local_files_only=os.path.isdir(model_path)).eval()
    tokenizer = AutoTokenizer.from_pretrained(source,local_files_only=os.path.isdir(model_path))

    if device.type != "cuda":
        model.to(device)

    return model, tokenizer

def get_batch_size(device):
    if device.type == "cpu": return min(32, os.cpu_count() or 4)
    if device.type == "mps": return 16
    free = torch.cuda.mem_get_info(device)[0] / 1024**3
    if free > 70: return 64
    if free > 40: return 32
    if free > 20: return 16
    if free > 10: return 8
    return 2

def select_device(requested=None):
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable (check `nvidia-smi` / driver version — an old driver "
              "vs. a newer PyTorch CUDA build is the most common cause); falling back to CPU")
        requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable"); requested = "cpu"
    device = torch.device(requested) if requested else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")
    elif device.type == "cpu":
        print(f"CPU cores available: {os.cpu_count()}")
    return device

def read_csv(input_path):
    df = pd.read_csv(input_path)
    required = {"tcr_id","TCRA","TCRB","MHC_seq","Epitope"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"CSV missing columns: {missing}")
    data = []
    for tid, a, b, mhc, epi in zip(df.tcr_id, df.TCRA, df.TCRB, df.MHC_seq, df.Epitope):
        if isinstance(a, str) and a: data.append((f"{tid}_TCRA", a.strip()))
        if isinstance(b, str) and b: data.append((f"{tid}_TCRB", b.strip()))
        if isinstance(mhc, str) and mhc: data.append((f"{tid}_MHC_seq", mhc.strip()))
        if isinstance(epi, str) and epi: data.append((f"{tid}_Epitope", epi.strip()))
    return data

def make_token_budget_batches(data, token_budget=4000):
    batches, current = [], []
    cur_max = 0
    for sid, seq in data:  # ya viene ordenado por longitud
        new_max = max(cur_max, len(seq))
        if current and new_max * (len(current) + 1) > token_budget:
            batches.append(current)
            current, cur_max = [], 0
            new_max = len(seq)
        current.append((sid, seq))
        cur_max = new_max
    if current:
        batches.append(current)
    return batches

def run_anarci_single(seq_id, seq):
    seq = seq.replace("*", "")
    try:
        results, _, _ = anarci([("seq", seq)], scheme="imgt", output=False)
    except Exception as e:
        print(f"[ANARCI EXCEPTION] {seq_id}: {e}")
        return seq_id, None

    if not results or results[0] is None:
        print(f"[ANARCI EMPTY] {seq_id}: sin resultado")
        return seq_id, None

    parsed = [(p, aa) for (p, _), aa in results[0][0][0] if aa != "-"]
    if not parsed:
        print(f"[ANARCI NO PARSED] {seq_id}")
        return seq_id, None

    cdrs = {}
    try:
        for name, (lo, hi) in cdr_ranges.items():
            s = "".join(aa for p, aa in parsed if lo <= p <= hi)
            start = seq.find(s)
            if start < 0:
                print(f"[ANARCI SUBSTRING FAIL] {seq_id} region={name} extracted={s!r}")
                return seq_id, None
            cdrs[name] = {"sequence": s, "start": start, "end": start + len(s) - 1}
        a = parsed[0][0] - 1
        b = parsed[-1][0] - 1
        cdrs["var"] = {"sequence": seq[a:b + 1], "start": a, "end": b}
    except Exception as e:
        print(f"[ANARCI CDR BUILD FAIL] {seq_id}: {e}")
        return seq_id, None

    return seq_id, cdrs

def precompute_anarci(data):
    seqs = [(sid, seq) for sid, seq in data if sid.endswith(("_TCRA", "_TCRB"))]
    if not seqs: return {}
    print(f"Running ANARCI on {len(seqs)} sequences")
    cache = {}
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        futures = [pool.submit(run_anarci_single, sid, seq) for sid, seq in seqs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="ANARCI"):
            sid, cdr = f.result()
            if cdr: cache[sid] = cdr
    return cache

def extract_batch(model, tokenizer, batch, anarci_cache, normalize, device):
    seq_ids = [x[0] for x in batch]
    seqs = [x[1] for x in batch]
    inputs = tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    with torch.inference_mode():
        hs = model(**inputs, output_hidden_states=True).hidden_states[-1]
        if normalize: hs = F.normalize(hs, dim=-1)
        hs = hs.cpu()
    results = {}
    for i, (sid, seq) in enumerate(batch):
        emb = hs[i]
        if sid in anarci_cache:
            out = {}
            for region, pos in anarci_cache[sid].items():
                s = pos["start"] + 1
                e = pos["end"] + 2
                out[region] = emb[s:e].mean(0, keepdim=True)
            results[sid] = out
        elif sid.endswith(("_TCRA", "_TCRB")):
            L = len(seq)
            results[sid] = {"var": emb[1:L + 1].mean(0, keepdim=True)}
        else:
            L = len(seq)
            results[sid] = {"": emb[1:L + 1].mean(0, keepdim=True)}
    return results

def append_to_h5(h5f, tensors):
    for sid, regions in tensors.items():
        for name, emb in regions.items():
            arr = emb.float().numpy()
            key = sid if name == "" else f"{sid}_{name}"
            if key in h5f:
                h5f[key][...] = arr
            else:
                h5f.create_dataset(key, data=arr, compression="gzip", chunks=True)

def main():
    parser = argparse.ArgumentParser(description="Generate ESMC embeddings")
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", default="embeddings.h5")
    parser.add_argument("-norm", "--normalized", action="store_true")
    parser.add_argument("-d", "--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile entirely (useful if it stalls or errors)")
    args = parser.parse_args()
 
    if not Path(args.input_path).exists():
        raise FileNotFoundError(args.input_path)
 
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
 
    t0 = time.time()
 
    device = select_device(args.device)
    batch_size = args.batch_size or get_batch_size(device)
 
    if args.no_compile:
        torch.compile = lambda m, *a, **k: m
 
    model, tokenizer = load_model(device)
 
    print(f"Reading {args.input_path}")
    data = read_csv(args.input_path)
    print(f"Loaded {len(data)} sequences")
 
    anarci_cache = precompute_anarci(data)
    data.sort(key=lambda x: len(x[1]))
    print(f"Processing {len(data)} sequences in batches of {batch_size}")
 
    batches = make_token_budget_batches(data, token_budget=batch_size * 200)

    with h5py.File(args.output_path, "w") as h5f:
        for batch in tqdm(batches, desc="Batches"):
            tensors = extract_batch(
                model,
                tokenizer,
                batch,
                anarci_cache,
                args.normalized,
                device,)
            append_to_h5(h5f, tensors)
            del tensors
 
    if device.type == "cuda": torch.cuda.empty_cache()
    print(f"Done in {time.time() - t0:.1f}s → {args.output_path}")
 
if __name__ == "__main__":
    main()
