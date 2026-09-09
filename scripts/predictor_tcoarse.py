#!/usr/bin/env python3
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import argparse
import pandas as pd
import argparse
import time

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

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Predict TCR-pMHC interactions using a trained XGBoost model.")
    parser.add_argument("-df","--input_df", type=str, required=True)
    parser.add_argument("-m","--model", type=str, required=True)
    parser.add_argument("-out","--output_path", type=str, required=True)
    args = parser.parse_args()

    model_path = args.model

    print(f"Loading model from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)

    path = args.input_df
    print(f"\nProcessing test set: {path}")

    hold_df = pd.read_csv(path, low_memory=False)
    lengths = hold_df["Epitope"].str.len().values

    energy_matrix = hold_df[[f"e_tcr_p{i}" for i in range(1, 14)]].values
    contact_matrix = hold_df[[f"c_tcr_p{i}" for i in range(1, 14)]].values

    energy_matrix_pmhc = hold_df[[f"e_pmhc_p{i}" for i in range(1, 14)]].values
    contact_matrix_pmhc = hold_df[[f"c_pmhc_p{i}" for i in range(1, 14)]].values

    e_aln = align_matrix(energy_matrix, lengths)
    c_aln = align_matrix(contact_matrix, lengths)
    
    e_aln_pmhc = align_matrix(energy_matrix_pmhc, lengths)
    c_aln_pmhc = align_matrix(contact_matrix_pmhc, lengths)
    
    for i in range(9):
        hold_df[f"e_tcr_p{i+1}_aln"] = e_aln[:, i]
        hold_df[f"c_tcr_p{i+1}_aln"] = c_aln[:, i]
        hold_df[f"e_pmhc_p{i+1}_aln"] = e_aln_pmhc[:, i]
        hold_df[f"c_pmhc_p{i+1}_aln"] = c_aln_pmhc[:, i]

    X_hold, y_hold = prepare_features(hold_df)

    #print("\nModel features:")
    #print(model.feature_names)
    #print("\nDF features:")
    #print(X_hold.columns.tolist())
    #print(f"\nModel n_features: {len(model.feature_names)}")
    #print(f"DF n_features: {X_hold.shape[1]}")
    
    hold_ids = X_hold.index
    dhold = xgb.DMatrix(X_hold)
    hold_preds = model.predict(dhold)
    
    df_preds = pd.DataFrame({"tcr_id": hold_ids,"Label": y_hold.values,"probs": hold_preds}).reset_index(drop=True)
    # Take for eahc tcr_id the TCRA, TCRB, MHC_allele, Epitope
    meta = hold_df.groupby("tcr_id", observed=True)[["TCRA", "TCRB", "MHC_allele", "Epitope"]].first().reset_index()
    df_preds = df_preds.merge(meta, on="tcr_id", how="left")
    df_preds = df_preds.sort_values("probs", ascending=False)

    out_path = args.output_path
    df_preds.to_csv(out_path, index=False)

    print("\n====================")
    print(f"Held-out set: {path}")

    # Score the predictions only when the input carried real ground truth of
    # both classes. prepare_features() fills in Label=0 when the column is
    # missing, which is the case when running inference on unlabelled
    # structures, and roc_auc_score() raises "Only one class present in y_true"
    # on a single-class column. Matches predictor_esmc / predictor_bimodal,
    # which already skip the metrics when the test CSV has no Label column.
    if "Label" not in hold_df.columns:
        print("No Label column in the input: skipping ROC AUC / PR AUC.")
    elif y_hold.nunique() < 2:
        print(
            f"Only one class present in Label ({y_hold.iloc[0]}): "
            "skipping ROC AUC / PR AUC."
        )
    else:
        roc_auc = roc_auc_score(y_hold, hold_preds)
        precision, recall, _ = precision_recall_curve(y_hold, hold_preds)
        pr_auc = auc(recall, precision)
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")

    print(f"Saved predictions to {out_path}")
    print("==================== \n")
    
    print("TOTAL TIME:", time.time() - t0)

if __name__ == "__main__":
    main()
