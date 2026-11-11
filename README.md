# TCoaRse

TCoaRse predicts TCR–peptide–MHC (TCR–pMHC) recognition using coarse-grained structural descriptors and ESMC protein language model embeddings. The pipeline prepares candidate complexes for AlphaFold3 (AF3), processes their predicted structures, and generates predictions with three pretrained models: TCoaRse-Str, TCoaRse-Seq and TCoaRse-Bi.

Run the commands below from the repository root. The examples use `example.csv` as input, `examples/` for AF3 outputs and `results/` for intermediate files. Configure the paths to external predictors, the ESMC checkpoint, the pretrained models, the energetic potentials and the pyDock Singularity container before running the corresponding steps.

## Part I. Prepare inputs for AlphaFold 3

### 1. Prepare the input CSV

| Component | Columns |
|---|---|
| TCR | `TCRA` and `TCRB` for full chain sequences, or `TRAV`, `TRAJ`, `TRBV`, `TRBJ`, `CDR3a` and `CDR3b` for sequence reconstruction |
| MHC class I (4 digit resolution e.g A*02:01) | `MHC_allele` or `MHC_seq` |
| Peptide | `Epitope` |
| Identifier | Optional `tcr_id`. The pipeline generates IDs when the column is absent. |

### 2. Prepare TCR–pMHC candidates

```bash
python scripts/stitch.py \
  --in_df ./example.csv \
  --out_f ./results \
  --predig_model neoant
```

`stitch.py` reconstructs TCR sequences with Stitchr/Thimble when needed, retrieves HLA sequences from `data/hla_prot.fasta`, and writes `results/tcrpmhc_full.csv`. It runs the following predictors.

| Predictor | Purpose |
|---|---|
| NetCleave | Peptide cleavage |
| NetMHCpan | Peptide–MHC binding |
| MHCflurry | Peptide processing, MHC binding and antigen presentation |
| PredIG | Peptide immunogenicity. Available modes are `neoant` for cancer antigens, `path` for pathogen-derived antigens and `noncan` for non-canonical cancer antigens. |
| NOAH | Peptide–MHC structural complementarity |

### 3. Filter candidate pMHCs

```bash
python scripts/pmhc_filtering.py \
  --out_f ./results \
  --tcrpmhcs ./results/tcrpmhc_full.csv
```

The script merges predictor outputs and writes `results/pmhc_scores.csv` and `results/tcrpmhc_filtered.csv`. Edit the `thresholds` dictionary in `pmhc_filtering.py` to customize the cutoffs. An example filter combines `NetCleave_strong`, `BA_rank_strong` and `NOAH_weak`.

| Metric | Strong cutoff | Weak cutoff |
|---|---:|---:|
| NetCleave | 0.6 | — |
| `mhcflurry_pres` | 1.0 | 2.0 |
| `mhcflurry_proc` | 1.0 | 2.0 |
| NOAH | −5 | −1 |
| PredIG | 0.19 | — |
| `EL_rank` | 0.5 | 2.0 |
| `mhcflurry_aff` | 1.0 | 2.0 |
| `BA_rank` | 0.5 | 2.0 |

### 4. Generate AF3 JSON files

```bash
python scripts/af3_jsons.py \
  --in_df ./results/tcrpmhc_filtered.csv \
  --out ./results/af3_jsons \
  --server
```

Use `--server` for AlphaFold Server. Omit it to generate inputs for a local AF3 installation. The downstream examples assume the following chain assignments.

| Chain | Component |
|---|---|
| A | MHC-I heavy chain |
| B | β2-microglobulin |
| C | Peptide |
| D | TCR α |
| E | TCR β |

## Part II. Predict structures with AlphaFold 3

Submit the JSON files to AlphaFold Server or run them through a local AF3 installation. Save the predicted structures and confidence files under `examples/` for the commands below. The example workflow uses five predicted models per complex and seed.

## Part III. Run downstream analyses

### 5. Collect PDB models and extract metadata

```bash
python scripts/cp_models.py ./examples/ ./results/examples_pdb --workers 16

python scripts/metadata_from_str.py ./results/examples_pdb \
  -o ./results/examples_metadata.csv
```

If you already have PDB files, start with `metadata_from_str.py` and point it at their directory. Name the files `{tcr_id}_{model_number}.pdb`, for example `XX_YY_ZZ_0.pdb`, and use that directory in subsequent commands. The PDB-only route supports energetic scoring, embedding generation and TCoaRse prediction. Structural quality assessment uses the original AF3 confidence files.

### 6. Assess structural quality (optional)

`process_folder.py` extracts AF3 confidence metrics for the complex and its interfaces, including pLDDT for the TCR CDRs, ipTM, interface PAE and interface PDE. It also calculates pDockQ, pDockQ2 and ipSAE. `quality_tier.py` uses the pretrained Random Forest in `rf_quality.pkl` to assign HQ, MQ, AQ or LQ.

```bash
python scripts/process_folder.py ./examples/ \
  --output ./results/examples_metrics.csv \
  --workers 16

python scripts/quality_tier.py \
  --test ./results/examples_metrics.csv \
  --output ./results/examples_quality_tiers.csv
```

### 7. Compare candidates with the AF3 training reference set (optional)

`similarities_af3.py` compares candidate sequences and structures with the configured AF3 training reference set. It uses TCRdist distances for individual TCR CDRs, normalized Smith–Waterman scores for MHC sequences, normalized Levenshtein similarity for peptides and DockQ interface metrics for structures.

```bash
python scripts/similarities_af3.py \
  -pdb_info results/examples_metadata.csv \
  -pdb_dir results/examples_pdb \
  -out_seq results/examples_af3_sim_seq.csv \
  -out_str results/examples_af3_sim_str.csv \
  -workers 16
```

The script takes the reference CSV and PDB paths from `PDB_INFO_CSV` and `PDB_DIR` in `src/config.py`. Override them with `-pre` and `-pre_pdb`. The current script appends `_merged` to the normalized `-pre_pdb` path, so point `-pre_pdb` at the base path. For example, `-pre_pdb data/pdbs` uses the precomputed files in `data/pdbs_merged/`. Expected names are `{tcr_id}_{model_number}_merged.pdb` when the reference CSV contains `model_number`, and `{tcr_id}_merged.pdb` otherwise.

### 8. Compare AF3 replicas (optional)

`pw_sim.py` calculates pairwise DockQ scores between models of the same complex to quantify structural agreement across replicas.

```bash
python scripts/pw_sim.py \
  --folder ./results/examples_pdb/ \
  --output ./results/examples_pairwise_dockq.csv \
  --workers 16
```

### 9. Generate interface contact maps

```bash
python scripts/contact_maps.py \
  -pdb results/examples_pdb \
  -out results/examples_cm \
  -workers 16 \
  -cm D:E:C:B:A
```

The chain map follows the order TCRα, TCRβ, peptide, β2-microglobulin and MHC heavy chain. The default is `D:E:C:B:A`. The script can use assignments from `data/chain_info.txt` for experimental structures listed there.

### 10. Calculate pyDock and TCoaRse energies

pyDock treats the TCR as the ligand and the pMHC as the receptor. Its runner prepares a manifest, validates chain assignments, runs chunks through the configured Singularity container and collects the results in an archive.

```bash
python scripts/run_pydock.py \
  --input results/examples_pdb \
  --output ./results/examples_pydock/

python scripts/energetic_scorer.py \
  -cmd results/examples_cm \
  -pdb results/examples_pdb \
  -pot pretrained_models/potentials/pot_7 \
  -chains D:E:C:B:A \
  -out results/examples_tcoarse_energies.csv \
  -t 7 \
  -w 16

python scripts/merge_energies.py \
  -tcoarse results/examples_tcoarse_energies.csv \
  -pydock results/examples_pydock/examples_pydock_ene.tar \
  -metadata results/examples_metadata.csv \
  -out results/examples_tcoarse_pydock_energies.csv
```

Use the actual archive path reported by `run_pydock.py` if its filename differs. The merged CSV supplies the structural and energetic features for TCoaRse-Str and TCoaRse-Bi.

### 11. Generate ESMC embeddings

```bash
python scripts/emb_generator.py \
  -i results/examples_metadata.csv \
  -o results/examples_emb.h5 \
  -d cuda \
  --no-compile
```

The HDF5 file contains pooled ESMC embeddings for the TCR chains, MHC and peptide. Use the embedding settings expected by the pretrained models. Add `--normalized` to normalize residue embeddings before pooling if the checkpoint was trained with that setting.

### 12. Run TCoaRse predictors

| Model | Inputs |
|---|---|
| TCoaRse-Str | Merged structural and energetic features |
| TCoaRse-Seq | Sequence metadata and ESMC embeddings |
| TCoaRse-Bi | Merged structural and energetic features and ESMC embeddings |

```bash
python scripts/predictor_tcoarse_str.py \
  -df results/examples_tcoarse_pydock_energies.csv \
  -m pretrained_models/tcoarse_str.json \
  -out results/examples_tcoarse_str_preds.csv

python scripts/predictor_tcoarse_seq.py \
  -df results/examples_metadata.csv \
  -emb results/examples_emb.h5 \
  -m pretrained_models/tcoarse_seq.json \
  -out results/examples_tcoarse_seq_preds.csv

python scripts/predictor_tcoarse_bi.py \
  -df results/examples_tcoarse_pydock_energies.csv \
  -emb results/examples_emb.h5 \
  -m pretrained_models/tcoarse_bi.json \
  -out results/examples_tcoarse_bi_preds.csv
```

### 13. Export final results

```bash
python scripts/generate_final_output.py \
  -metadata results/examples_metadata.csv \
  -tcoarse_str results/examples_tcoarse_str_preds.csv \
  -tcoarse_seq results/examples_tcoarse_seq_preds.csv \
  -tcoarse_bi results/examples_tcoarse_bi_preds.csv \
  -pydock results/examples_tcoarse_pydock_energies.csv \
  -pw results/examples_pairwise_dockq.csv \
  -sim_seq results/examples_af3_sim_seq.csv \
  -sim_str results/examples_af3_sim_str.csv \
  -qm results/examples_quality_tiers.csv \
  -o final_results
```

The current exporter requires the files from steps 7 and 8. If you skip those analyses, use the individual prediction CSVs from step 12. The example supplies `-qm` because versions that read the quality CSV unconditionally fail when it is omitted. If your version reads and exports quality metrics conditionally, omit `-qm` when skipping step 6.

| Output | Contents |
|---|---|
| `tcoarse_all.csv` | Outer join of the three prediction tables by `tcr_id`. Missing values remain empty. |
| `tcoarse_predictions.csv` | Combined predictions with metadata IDs added where missing and missing values filled with zero. These zeros can represent missing predictions. |
| `metadata.csv` | TCR–pMHC metadata |
| `model_quality_metrics.csv` | Quality table supplied with `-qm` |
| `pydock_energies.csv` | pyDock scores by TCR and model |
| `replica_consensus_analysis.csv` | Pairwise DockQ results across replicas |
| `af3_similarity_analysis_seq.csv` | Minimum TCRdist distances and maximum MHC and epitope similarities |
| `af3_similarity_analysis_str.csv` | Maximum DockQ and Fnat and minimum iRMSD, LRMSD and clashes. Each extreme is selected independently. |

## Run the pipeline with Nextflow

Run `downstream.nf` from the repository root after AlphaFold 3 has generated the structures. Use `./nextflow` if the launcher is in the repository root, or `nextflow` if it is on your `PATH`.

- Enable all optional analyses with `--quality --similarity --consensus`. These run structural quality assessment, comparison with AF3 references and pairwise DockQ between replicas, respectively.
- For the TCoaRse predictions and their required preprocessing, omit the three flags or pass `--quality false --similarity false --consensus false`.
- Pass `--pydock_archive` and/or `--embeddings_h5` to reuse precomputed files for the same inputs. The pipeline runs the corresponding generation step for each file you omit.

The example enables all optional analyses and reuses both precomputed files. Intermediate outputs go to `results/`, final CSVs go to `final_results/`, and `-resume` reuses eligible cached tasks from an earlier run.

```bash
./nextflow run downstream.nf \
  --af3_dir examples \
  --quality --similarity --consensus \
  --results_dir results \
  --outdir final_results \
  --pydock_archive results/examples_pydock_ene.tar \
  --embeddings_h5 results/examples_emb.h5 \
  -resume
```

Replace the example archive and HDF5 paths with the locations of your precomputed files.

## Repository structure

| Path | Contents |
|---|---|
| `scripts/` | Input preparation, structure processing, scoring, prediction and export scripts |
| `pretrained_models/` | TCoaRse model JSON files, the quality classifier and energetic potentials |
| `data/` | HLA sequences, chain annotations and AF3 reference data |
| `src/` | Configuration and structural confidence utilities, including pDockQ, pDockQ2 and ipSAE |
| `pydock/` | pyDock configuration and worker scripts |
