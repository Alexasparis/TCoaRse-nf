#!/usr/bin/env python3
import os
import argparse
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-metadata","--file_metadata",required=True)
    ap.add_argument("-qm","--file_metrics",required=True)
    ap.add_argument("-tcoarse_str","--file_tcoarse_str",required=True)
    ap.add_argument("-tcoarse_seq","--file_tcoarse_seq",required=True)
    ap.add_argument("-tcoarse_bi","--file_tcoarse_bi",required=True)
    ap.add_argument("-pydock","--file_pydock",required=True)
    ap.add_argument("-pw","--file_consensus",required=True)
    ap.add_argument("-sim_seq","--file_similarities_seq",required=True)
    ap.add_argument("-sim_str","--file_similarities_str",required=True)
    ap.add_argument("-o","--output_folder",required=True)
    a=ap.parse_args()

    # Create output folder if it does not exist
    if not os.path.exists(a.output_folder):
        os.makedirs(a.output_folder)

    # Open csvs 
    metadata = pd.read_csv(a.file_metadata)
    metrics = pd.read_csv(a.file_metrics)
    tcoarse_str = pd.read_csv(a.file_tcoarse_str)
    tcoarse_seq = pd.read_csv(a.file_tcoarse_seq)
    tcoarse_bi = pd.read_csv(a.file_tcoarse_bi)
    pydock = pd.read_csv(a.file_pydock)
    consensus = pd.read_csv(a.file_consensus)
    similarities_seq = pd.read_csv(a.file_similarities_seq)
    similarities_str = pd.read_csv(a.file_similarities_str)

    # In similarities seq keep cols starting with tcrdist and ending with min, and cols starting with sim and ending with max
    similarities_seq = similarities_seq[['tcr_id'] + [col for col in similarities_seq.columns if col.startswith('tcrdist') and col.endswith('min')] + [col for col in similarities_seq.columns if col.startswith('sim') and col.endswith('max')]]
    similarities_str = similarities_str[['tcr_id'] + ['model_number'] + [col for col in similarities_str.columns if col.startswith('DockQ') and col.endswith('max')] + [col for col in similarities_str.columns if col.startswith('iRMSD') and col.endswith('min')]
                                        + [col for col in similarities_str.columns if col.startswith('LRMSD') and col.endswith('min')] + [col for col in similarities_str.columns if col.startswith('Fnat') and col.endswith('max')]
                                        + [col for col in similarities_str.columns if col.startswith('clashes') and col.endswith('min')]]

    # From pydock filter tcr_id, model_number, Conf,Ele,Desolv,VDW,Total
    pydock = pydock[['tcr_id', 'model_number', 'Conf', 'Ele', 'Desolv', 'VDW', 'Total']]

    # In Tcoarse rename probs to tcoarse_str, tcoarse_seq, tcoarse_bi
    tcoarse_str = tcoarse_str.rename(columns={'probs': 'tcoarse_str'})
    tcoarse_seq = tcoarse_seq.rename(columns={'probs': 'tcoarse_seq'})
    tcoarse_bi = tcoarse_bi.rename(columns={'probs': 'tcoarse_bi'})

    # Generate a tcoarse file with all predictions
    tcoarse_all = pd.merge(tcoarse_str, tcoarse_seq, on=['tcr_id'], how='outer')
    tcoarse_all = pd.merge(tcoarse_all, tcoarse_bi, on=['tcr_id'], how='outer')
    for tcr_id in metadata['tcr_id']:
        if tcr_id not in tcoarse_all['tcr_id'].values:
            tcoarse_all = pd.concat([tcoarse_all, pd.DataFrame({'tcr_id': [tcr_id]})], ignore_index=True)
    tcoarse_all = tcoarse_all.fillna(0)
    tcoarse_all.to_csv(os.path.join(a.output_folder, "tcoarse_predictions.csv"), index=False)

    # Move all files to the output folder with a different name
    metadata.to_csv(os.path.join(a.output_folder, "metadata.csv"), index=False)
    metrics.to_csv(os.path.join(a.output_folder, "model_quality_metrics.csv"), index=False)
    pydock.to_csv(os.path.join(a.output_folder, "pydock_energies.csv"), index=False)
    consensus.to_csv(os.path.join(a.output_folder, "replica_consensus_analysis.csv"), index=False)
    similarities_seq.to_csv(os.path.join(a.output_folder, "af3_similarity_analysis_seq.csv"), index=False)
    similarities_str.to_csv(os.path.join(a.output_folder, "af3_similarity_analysis_str.csv"), index=False)

if __name__ == "__main__":
    main()