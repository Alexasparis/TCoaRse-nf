#!/usr/bin/env python3
import os
import pandas as pd
import xgboost as xgb
import argparse
import time
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

_REQUIRED_SUFFIXES = {
    "var":   ["_TCRA_var",  "_TCRB_var",  "_Epitope", "_MHC_seq"],
    "cdrs":  ["_TCRA_cdr1", "_TCRA_cdr2", "_TCRA_cdr3",
              "_TCRB_cdr1", "_TCRB_cdr2", "_TCRB_cdr3",
              "_Epitope",   "_MHC_seq"],
    "cdr3s": ["_TCRA_cdr3", "_TCRB_cdr3", "_Epitope",  "_MHC_seq"],
    "cdr3a": ["_TCRA_cdr3", "_Epitope",   "_MHC_seq"],
    "cdr3b": ["_TCRB_cdr3", "_Epitope",   "_MHC_seq"],}

_HDF5_READ_WORKERS = min(16, (os.cpu_count() or 4) * 2)

contact_cols_tcrp_aln = [f'c_tcr_p{i}_aln' for i in range(1, 10)]
energy_cols_tcrp_aln = [f'e_tcr_p{i}_aln' for i in range(1, 10)]
ratios_tcrp = [f"ratio_{i}" for i in range(1, 10)]
ratios_tcrp_total = ['ratio_tcrp_total']

contact_cols_pmhc_aln = [f'c_pmhc_p{i}_aln' for i in range(1, 10)]
energy_cols_pmhc_aln = [f'e_pmhc_p{i}_aln' for i in range(1, 10)]
ratios_pmhc = [f"ratio_pmhc_p_{i}" for i in range(1, 10)]
ratios_pmhc_total = ['ratio_pmhc_total']

contact_cols_tcrmhc = ['c_tcr_mhc']
energy_cols_tcrmhc = ['e_tcr_mhc']
ratios_tcrmhc = ['ratio_mhc']

pydock_cols = ['Ele', 'Desolv', 'VDW', "Total"] # Total

def align_matrix(values, lengths):
    out = np.zeros((len(values), 9), dtype=np.float32)
    for idx in range(len(values)):
        original = values[idx][:lengths[idx]]
        out[idx, 0], out[idx, 1], out[idx, 2], out[idx, 8] = original[0], original[1], original[2], original[-1]
        center = original[3:-1]
        if len(center) == 5:
            out[idx, 3:8] = center
        elif len(center) < 5:
            indices = np.linspace(0, len(center)-1, 5)
            out[idx, 3:8] = [center[int(round(i))] for i in indices]
        else:
            out[idx, 3:8] = [chunk.mean() for chunk in np.array_split(center, 5)]
    return out

def prepare_features(df):
    df = df.copy()
    if "Label" not in df.columns:
        df["Label"] = 0
    for i in range(1, 10):
        df[ratios_tcrp[i-1]] = -df[f'e_tcr_p{i}_aln'] / (df[f'c_tcr_p{i}_aln'] + 1e-8)
        df[ratios_pmhc[i-1]] = -df[f'e_pmhc_p{i}_aln'] / (df[f'c_pmhc_p{i}_aln'] + 1e-8)
    df["e_pmhc_p_all"] = df[[f"e_pmhc_p{i}_aln" for i in range(1, 10)]].sum(axis=1)
    df["c_pmhc_p_all"] = df[[f"c_pmhc_p{i}_aln" for i in range(1, 10)]].sum(axis=1)
    df["ratio_tcrp_total"] = -df['e_tcr_p_all'] / (df['c_tcr_p_all'] + 1e-8)
    df["ratio_mhc"] = -df['e_tcr_mhc'] / (df['c_tcr_mhc'] + 1e-8)
    df["ratio_pmhc_total"] = -df['e_pmhc_p_all'] / (df['c_pmhc_p_all'] + 1e-8)
    if "pydock_cols" in globals() and pydock_cols is not None:
        pydock_cols_used = [c for c in pydock_cols if c in df.columns]
    else:
        pydock_cols_used = []
    #pydock_peptide=[f"{term}_p{i}_aln" for term in ["ele","vdw","total"] for i in range(1,10)]
    feat_cols = (
        contact_cols_tcrp_aln +
        #contact_cols_pmhc_aln +
        contact_cols_tcrmhc +
        ratios_tcrp +
        #ratios_pmhc +
        ratios_tcrmhc +
        #['c_tcr_p_all'] +
        #["c_pmhc_p_all"] +
        #ratios_tcrp_total +
        #ratios_pmhc_total +
        pydock_cols_used
        #+ model_quality_metrics
        #+ pydock_peptide
        )
    agg = df.groupby("tcr_id", observed=True)[feat_cols].agg(["mean", "min", "max", "std"])
    agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns]
    y = df.groupby("tcr_id", observed=True)["Label"].first()
    return agg.fillna(0).astype(np.float32), y

def _load_key(file_path: str, key: str):
    """Worker: open the HDF5 file and read a single dataset."""
    with h5py.File(file_path, "r") as f:
        return key, f[key][:]
    
def load_embeddings(file_path: str, data_type: str | None = None) -> dict:
    """
    Load embeddings from HDF5 with optional filtering + always-on tqdm progress bar.
    """

    required_suffixes = _REQUIRED_SUFFIXES.get(data_type)
    with h5py.File(file_path, "r") as f:
        all_keys = list(f.keys())

    if required_suffixes is not None:
        keys_to_load = [
            k for k in all_keys
            if any(k.endswith(s) for s in required_suffixes)]
        skipped = len(all_keys) - len(keys_to_load)

        print(
            f"Filtered HDF5: {len(keys_to_load)}/{len(all_keys)} keys "
            f"({skipped} skipped, data_type='{data_type}')"
        )
    else:
        keys_to_load = all_keys

    embeddings = {}

    with tqdm(total=len(keys_to_load), desc="Loading embeddings") as pbar:
        with ThreadPoolExecutor(max_workers=_HDF5_READ_WORKERS) as pool:
            futures = {
                pool.submit(_load_key, file_path, k): k
                for k in keys_to_load}

            for fut in as_completed(futures):
                key, arr = fut.result()
                embeddings[key] = arr
                pbar.update(1)

    return embeddings

def process_data_for_type(data_dict: dict, data_type: str, dataframe: pd.DataFrame):
    # Structural features
    print("Fast alignment...")
    lengths = dataframe["Epitope"].str.len().values
    
    energy_matrix = dataframe[[f"e_tcr_p{i}" for i in range(1, 14)]].values
    contact_matrix = dataframe[[f"c_tcr_p{i}" for i in range(1, 14)]].values

    energy_matrix_pmhc = dataframe[[f"e_pmhc_p{i}" for i in range(1, 14)]].values
    contact_matrix_pmhc = dataframe[[f"c_pmhc_p{i}" for i in range(1, 14)]].values

    e_aln = align_matrix(energy_matrix, lengths)
    c_aln = align_matrix(contact_matrix, lengths)

    e_aln_pmhc = align_matrix(energy_matrix_pmhc, lengths)
    c_aln_pmhc = align_matrix(contact_matrix_pmhc, lengths)

    for i in range(9):
        dataframe[f"e_tcr_p{i+1}_aln"], dataframe[f"c_tcr_p{i+1}_aln"] = e_aln[:, i], c_aln[:, i]
        dataframe[f"e_pmhc_p{i+1}_aln"], dataframe[f"c_pmhc_p{i+1}_aln"] = e_aln_pmhc[:, i], c_aln_pmhc[:, i]

    agg_structural, y = prepare_features(dataframe)
    id_to_row = {str(row["tcr_id"]): row for _, row in dataframe.iterrows()}
    suffixes = _REQUIRED_SUFFIXES.get(data_type, [])

    # Tcr_ids as str in all and as a col
    agg_structural.index = agg_structural.index.astype(str)
    dataframe["tcr_id"] = dataframe["tcr_id"].astype(str)

    # Extract unique TCR IDs from embedding keys
    tcr_id_set = set()
    for key in data_dict:
        for suf in suffixes:
            if key.endswith(suf):
                tcr_id_set.add(key[: -len(suf)])
                break

    # Output containers
    embeddings = { "tcra": [],"tcrb": [],"epitope": [],"mhc": [],"structural": []}
    tcr_ids = []
    labels_list = []
    fold_list = []

    # Main loop
    for tcr_id in sorted(tcr_id_set):
        print(f"Processing TCR ID: {tcr_id}", end="\r")
        row = id_to_row.get(tcr_id)
        if row is None:
            print(f"[WARNING] No matching row for {tcr_id}, skipping.")
            continue

        required = [tcr_id + s for s in suffixes]
        missing = [k for k in required if k not in data_dict]

        if missing:
            print(f"[WARNING] Missing keys for {tcr_id}: {missing}, skipping.")
            continue
        # Structural features check
        if tcr_id not in agg_structural.index:
            print(f"[WARNING] No structural features for {tcr_id}, skipping.")
            continue

        struct_feat = agg_structural.loc[tcr_id].values.astype(np.float32)

        if np.isnan(struct_feat).any():
            print(f"[WARNING] NaNs in structural features for {tcr_id}, skipping.")
            continue

        # Build embeddings
        if data_type == "var":

            tcra = data_dict[f"{tcr_id}_TCRA_var"]
            tcrb = data_dict[f"{tcr_id}_TCRB_var"]

        elif data_type == "cdrs":
            tcra = np.concatenate([data_dict[f"{tcr_id}_TCRA_cdr1"],data_dict[f"{tcr_id}_TCRA_cdr2"],data_dict[f"{tcr_id}_TCRA_cdr3"]], axis=1)
            tcrb = np.concatenate([data_dict[f"{tcr_id}_TCRB_cdr1"],data_dict[f"{tcr_id}_TCRB_cdr2"],data_dict[f"{tcr_id}_TCRB_cdr3"]], axis=1)

        elif data_type == "cdr3s":
            tcra = data_dict[f"{tcr_id}_TCRA_cdr3"]
            tcrb = data_dict[f"{tcr_id}_TCRB_cdr3"]

        elif data_type == "cdr3b":
            tcra = None
            tcrb = data_dict[f"{tcr_id}_TCRB_cdr3"]

        elif data_type == "cdr3a":
            tcra = data_dict[f"{tcr_id}_TCRA_cdr3"]
            tcrb = None

        else:
            continue

        # NaN checks
        arrays_to_check = [a for a in (tcra,tcrb,data_dict[f"{tcr_id}_Epitope"],data_dict[f"{tcr_id}_MHC_seq"])if a is not None]

        if any(np.isnan(a).any() for a in arrays_to_check):
            print(f"[WARNING] NaNs in embeddings for {tcr_id}, skipping.")
            continue

        # Store embeddings
        if tcra is not None:
            embeddings["tcra"].append(tcra)
        if tcrb is not None:
            embeddings["tcrb"].append(tcrb)

        embeddings["epitope"].append(data_dict[f"{tcr_id}_Epitope"])
        embeddings["mhc"].append(data_dict[f"{tcr_id}_MHC_seq"])
        embeddings["structural"].append(struct_feat.reshape(1, -1))

        tcr_ids.append(tcr_id)

        if "Fold" in dataframe.columns:
            fold_list.append(int(row["Fold"]))
        if "Label" in dataframe.columns:
            labels_list.append(row["Label"])

    print("\n=== Embedding shapes ===")
    for name, emb_list in embeddings.items():
        if len(emb_list) == 0:
            print(f"{name:12s}: EMPTY")
        else:
            arr = np.asarray(emb_list)
            print(f"{name:12s}: {arr.shape}")
    print(f"tcr_ids     : {len(tcr_ids)}")
    print(f"labels      : {len(labels_list)}")
    print(f"folds       : {len(fold_list)}")
    print("========================\n")
    return embeddings, tcr_ids, fold_list, labels_list

def build_X_matrix(embeddings_dict: dict, data_type: str) -> np.ndarray:
    n = len(embeddings_dict["epitope"])
    if n == 0:
        raise ValueError("No samples in embeddings_dict['epitope'].")

    _BOTH_CHAINS_REQUIRED = {"var", "cdrs", "cdr3s"}

    def stack(lst):
        """Stack a list of (1, D) arrays into (n, D); returns None if empty."""
        return np.vstack(lst) if lst else None

    tcra_arr = stack(embeddings_dict.get("tcra"))
    tcrb_arr = stack(embeddings_dict.get("tcrb"))
    epi_arr = stack(embeddings_dict["epitope"])
    mhc_arr = stack(embeddings_dict["mhc"])
    structural_arr = stack(embeddings_dict.get("structural"))

    # Validate required embeddings
    if epi_arr is None:
        raise ValueError("Epitope embeddings cannot be None or empty!")
    if mhc_arr is None:
        raise ValueError("MHC embeddings cannot be None or empty!")
    if structural_arr is None:
        raise ValueError("Structural embeddings cannot be None or empty!")

    if data_type in _BOTH_CHAINS_REQUIRED and (tcra_arr is None or tcrb_arr is None):
        raise ValueError(
            f"data_type='{data_type}' requires both TCRA and TCRB embeddings, "
            f"but got: TCRA={'present' if tcra_arr is not None else 'MISSING'}, "
            f"TCRB={'present' if tcrb_arr is not None else 'MISSING'}")

    parts = [a for a in (tcra_arr,tcrb_arr,epi_arr,mhc_arr,structural_arr) if a is not None]
    X = np.hstack(parts)
    print(f"Final feature matrix shape: {X.shape}")
    return X

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Predict TCR-pMHC interactions using a trained XGBoost model.")
    parser.add_argument("-df","--input_df", type=str, required=True)
    parser.add_argument("-emb","--embeddings", type=str, required=True)
    parser.add_argument("-m","--model", type=str, required=True)
    parser.add_argument("-out","--output_path", type=str, required=True)
    args = parser.parse_args()

    # Load model
    model_path = args.model
    print(f"Loading model from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    type_seq = "cdr3s"
    
    # ── TEST inference (aligned CSV + HDF5 per index) ────────
    emb_path = args.embeddings
    print(f"\nRunning inference on {len(args.input_df)} with embeddedings {emb_path}")
    print(f"Loading model from {model_path}")
    raw_embeddings_test = load_embeddings(emb_path, data_type=type_seq)
    test_df = pd.read_csv(args.input_df, low_memory=False)
    test_embeddings, test_ids, _, _ = process_data_for_type(raw_embeddings_test, type_seq, test_df)
    X_test = build_X_matrix(test_embeddings, type_seq)
    dtest = xgb.DMatrix(X_test)

    preds = model.predict(dtest)
    out = pd.DataFrame({"tcr_id": test_ids,"probs": preds}).sort_values("probs", ascending=False)
    out_file_path = args.output_path
    out.to_csv(out_file_path, index=False)

    # Print ROC AUC and PR AUC if true labels are available in the test CSV
    if "Label" in test_df.columns:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        # Tcr_id as str in both dataframes to avoid merge issues
        test_df["tcr_id"] = test_df["tcr_id"].astype(str)
        out["tcr_id"] = out["tcr_id"].astype(str)
        merged_test = pd.merge(test_df[["tcr_id", "Label"]],out,on="tcr_id",how="left").dropna(subset=["probs"])
        if merged_test.empty:
            print("No overlapping tcr_id between test CSV and predictions for AUC calculation.")
        elif merged_test["Label"].nunique() < 2:
            # roc_auc_score raises "Only one class present in y_true" here
            print("Only one class present in Label: skipping ROC AUC / PR AUC.")
        else:
            auc_roc = roc_auc_score(merged_test["Label"], merged_test["probs"])
            pr, rec, _ = precision_recall_curve(merged_test["Label"], merged_test["probs"])
            auc_pr = auc(rec, pr)
            print(f"ROC AUC: {auc_roc:.4f} | PR AUC: {auc_pr:.4f}")

    print(f"Saved -> {out_file_path}")
    
    print(f"\nTotal time: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()
