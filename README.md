# TCoaRse Pipeline

TCoaRse is a computational pipeline for TCR–pMHC pairing analysis that combines peptide–MHC binding prediction, AlphaFold3 (AF3) structure prediction, structural quality assessment, interface characterization, energetic analysis, and machine-learning-based TCR–pMHC pairing prediction.

The pipeline is divided into three main stages:

1. **Input preparation and AF3 setup:** prepares TCR–pMHC candidates, evaluates peptide–MHC binding, processing and cleavage, applies filtering criteria, and generates AF3 input JSON files.
2. **Structural modelling using AF3:** predicts TCR–pMHC complex structures using AlphaFold3.
3. **Downstream structural and predictive analysis:** processes AF3 structures, evaluates structural quality, extracts structural and energetic descriptors, calculates sequence and structural similarities to the AF3 training set, generates ESMC protein language model embeddings, and produces TCoaRse-Str, TCoaRse-Seq and TCoaRse-Bi predictions.

---

## Pipeline overview

```text
INPUT PREPARATION
      |
      v
Input CSV
      |
      v
stitch.py
  ├── Prepare TCR–pMHC data
  ├── Reconstruct TCR sequences
  ├── Add MHC sequences
  └── Run pMHC predictors
      |
      v
pmhc_filtering.py
  ├── Merge prediction results
  └── Filter candidate pMHCs
      |
      v
af3_jsons.py
  └── Generate AF3 JSON files
      |
      v
AlphaFold3
  └── TCR–pMHC structure prediction
      |
      v
DOWNSTREAM ANALYSIS
      |
      ├── Structural quality assessment
      |
      ├── Consensus analysis across five AF3 replicas using DockQ
      |
      ├── Similarity analysis to AF3 pMHC data
      |     ├── Sequence similarity
      |     └── Structural similarity
      |
      ├── Interface and energetic analysis
      |     ├── Contact maps
      |     ├── TCoaRse energetic scoring
      |     └── pyDock energetic scoring
      |
      ├── Protein language model embeddings
      |     └── ESMC embeddings
      |
      └── Machine-learning predictions
            ├── TCoaRse
            ├── BiModal
            └── ESMC
      |
      v
FINAL RESULTS
```

---

# Part I. Input preparation and AF3 structure generation

## 1. Input CSV

The input CSV must contain the following information:

- **TCR:** either `TRAV`, `TRAJ`, `TRBV`, `TRBJ`, `CDR3a`, `CDR3b`, or the complete `TCRA` and `TCRB` sequences.
- **MHC:** either `MHC_allele` or `MHC_seq`.
- **Epitope:** `Epitope`.
- **tcr_id:** optional identifier, automatically generated if missing.

## 2. Prepare TCR–pMHC data

```bash
python stitch.py \
    --in_df ../input.csv \
    --out_f ../out_f \
    --predig_model "neoant"
```

`stitch.py` prepares the TCR–pMHC input data.

**Main operations:**

- Reconstructs TCR sequences using **Stitchr/Thimble**, when necessary.
- Retrieves MHC sequences from `data/hla_prot.fasta`.
- Evaluates peptide cleavage, processing, MHC binding, presentation, and immunogenicity using:
  - **NetCleave:** peptide cleavage prediction.
  - **NetMHCpan:** peptide–MHC binding prediction.
  - **MHCflurry:** peptide processing, peptide–MHC binding, and antigen presentation prediction.
  - **PredIG:** peptide immunogenicity prediction, with three prediction modes:
    - `neoant`: cancer antigens.
    - `path`: pathogen-derived antigens.
    - `noncan`: non-canonical cancer antigens.
  - **NOAH:** peptide–MHC structural complementarity prediction.

**Output:**

- `tcrpmhc_full.csv`

## 3. Filter candidate pMHCs

```bash
python pmhc_filtering.py \
    --out_f ../out_f \
    --tcrpmhcs ../out_f/tcrpmhc_full.csv
```

Prediction results are merged and candidate pMHCs are filtered using predefined or custom filtering criteria.

Example filtering configuration:

`NetCleave_strong + BA_rank_strong + NOAH_weak`

**Outputs:**

- `pmhc_scores.csv`
- `tcrpmhc_filtered.csv`

## 4. Generate AlphaFold3 inputs

```bash
python af3_jsons.py \
    --in_df ../out_f/tcrpmhc_filtered.csv \
    --out ../out_f/af3_jsons
```

Generates the AF3 input JSON files required for TCR–pMHC class I structure prediction.

Each complex contains five chains:

| Chain | Component |
|---|---|
| A | MHC-I |
| B | β2-microglobulin |
| C | Peptide |
| D | TCR α |
| E | TCR β |

---

# Part II. Structural modelling

## 5. AlphaFold3 structure prediction

The generated JSON files are submitted to **AlphaFold3** for structure prediction.

AF3 produces the predicted TCR–pMHC structures and associated confidence metrics, with five replicas per complex and seed.

The resulting AF3 structures are used as input for the downstream analysis.

---

# Part III. Downstream structural and predictive analysis

## 6. Downstream analysis scripts

The downstream analysis consists of a collection of Python scripts that process the AF3 output structures and perform structural quality assessment, metadata extraction, similarity analysis, interface characterization, energetic calculations, embedding generation, and machine-learning-based predictions.

The scripts are executed according to their dependencies, progressively generating the structural and energetic descriptors required by the TCoaRse, BiModal and ESMC prediction models.

### 6.1. Configuration and input files

The downstream analysis requires the following input files and resources:

| Input | Description |
|---|---|
| `af3_outputs` | Directory containing the AF3 predictions |
| `potential` | TCoaRse energetic potential files |
| `tcoarse.json` | Pretrained TCoaRse model |
| `bimodal.json` | Pretrained BiModal model |
| `esmc.json` | Pretrained ESMC model |
| `rf_quality.pkl` | Pretrained Random Forest quality classification model |
| `af3_training.csv` | Reference AF3 training metadata |
| `pdbs/` | Reference PDB structures |
| `scripts/` | Directory containing the Python scripts |
| `pdockq.py` | pDockQ calculation script |
| `pdockq2_pae.py` | pDockQ2 calculation script |
| `ipsae.py` | ipSAE calculation script |

The output basename is derived from the AF3 output directory name. For example, an input directory named `results` generates output files with the prefix `results`.

For the examples below, assume that the AF3 predictions are stored in `examplex/` and that commands are executed from the repository root. Replace `examplex/` with the path to your own AF3 output directory. The commands illustrate the intended input/output workflow; check each script's command-line options if your local version requires additional arguments.

### 6.2. Structural quality assessment

**Scripts:** `process_folder.py`, `quality_tier.py`

The first step evaluates the structural quality and confidence of the AF3-predicted TCR–pMHC complexes.

`process_folder.py` processes the AF3 output directory and calculates a comprehensive set of structural confidence and interface quality metrics, including:

- **pLDDT:** evaluates local structural confidence across the full complex and individual TCR CDR regions (CDR1, CDR2 and CDR3) of both the TCR α and β chains.
- **ipTM:** evaluates inter-chain structural confidence, including the mean ipTM and the ipTM specifically assessing the TCR–pMHC interfaces.
- **iPAE:** interface predicted aligned error, scaled and normalized to a 0–1 range, where 0 indicates lower confidence and 1 indicates higher confidence.
- **iPDE:** interface predicted distance error/contact probabilities, ranging from 0 to 1, describing the confidence of predicted inter-residue contacts.
- **pDockQ** and **pDockQ2:** estimate the confidence of predicted protein–protein interfaces.
- **ipSAE:** refined ipTM using PAE scores.

`quality_tier.py` subsequently applies a pretrained Random Forest model (`rf_quality.pkl`) to classify the predicted complexes into four structural quality tiers based on the calculated metrics.

**Example using `examplex/`:**

```bash
python scripts/process_folder.py examplex/
python scripts/quality_tier.py examplex_metrics.csv
```

**Output:**

- `<basename>_metrics.csv`

### 6.3. PDB extraction and metadata generation

**Scripts:** `cp_models.py`, `metadata_from_str.py`

`cp_models.py` collects and copies the predicted AF3 structures into a single PDB directory.

`metadata_from_str.py` extracts the relevant metadata from the PDB structures, including the information required for downstream sequence, structural and energetic analyses.

**Example using `examplex/`:**

```bash
python scripts/cp_models.py examplex/
python scripts/metadata_from_str.py examplex_pdb/
```

**Outputs:**

- `<basename>_pdb/`
- `<basename>_metadata.csv`

### 6.4. Sequence and structural similarity analysis

**Script:** `similarities_af3.py`

Calculates the similarity between newly predicted TCR–pMHC complexes and a reference AF3 training dataset.

The analysis compares:

- **Sequence similarity:** evaluates sequence-level similarity between reference and newly predicted complexes.
- **Structural similarity:** evaluates structural similarity between reference and newly predicted complexes.

Reference data are provided through `af3_training.csv` and the corresponding reference PDB directory.

**Example using `examplex/`:**

```bash
python scripts/similarities_af3.py examplex_metadata.csv
```

**Outputs:**

- `sim_seq.csv`
- `sim_str.csv`

### 6.5. pyDock energetic analysis

**Scripts:** `01_make_manifest.py`, `02_make_chunks.py`, `03_validate_chain_mapping.py`, `worker_chunk.py`

The pyDock module calculates docking-related energetic descriptors for the predicted TCR–pMHC complexes.

The process consists of:

1. Generating the input manifest.
2. Splitting the input structures into computational chunks.
3. Validating chain mapping.
4. Executing the pyDock worker using the configured Singularity container.
5. Collecting the resulting energetic data into an archive.

**Example workflow for `examplex/` (run from the repository root):**

```bash
python pydock/01_make_manifest.py
python pydock/02_make_chunks.py
python pydock/03_validate_chain_mapping.py
python pydock/worker_chunk.py
```

The input directory and output locations for this module are configured through the pyDock manifest/configuration files.

**Output:**

- `<basename>_pydock_ene.tar`

### 6.6. Interface contact maps

**Script:** `contact_maps.py`

Generates residue-level contact maps for the predicted TCR–pMHC structures.

These contact maps describe the interactions between the TCR and pMHC components and provide the structural information required for subsequent energetic scoring.

**Example using `examplex/`:**

```bash
python scripts/contact_maps.py examplex_pdb/
```

**Output:**

- `<basename>_cm/`

### 6.7. Pairwise structural similarity

**Script:** `pw_sim.py`

Calculates pairwise DockQ similarity between the five models of a given predicted TCR–pMHC complex.

This analysis provides a measure of structural agreement across different predicted models and can be used to assess structural consistency.

**Example using `examplex/`:**

```bash
python scripts/pw_sim.py examplex_pdb/
```

**Output:**

- `<basename>_pairwise_dockq.csv`

### 6.8. TCoaRse energetic scoring

**Script:** `energetic_scorer.py`

Calculates energetic descriptors from the predicted structures using the generated contact maps and pretrained TCoaRse energetic potentials.

The calculation uses:

- Predicted PDB structures.
- Interface contact maps.
- Pretrained energetic potentials.

The resulting descriptors characterize the energetic interactions within the TCR–pMHC interface.

**Example using `examplex/`:**

```bash
python scripts/energetic_scorer.py examplex_pdb/ examplex_cm/
```

**Output:**

- `<basename>_tcoarse_energies.csv`

### 6.9. Merge energetic descriptors

**Script:** `merge_energies.py`

Combines the TCoaRse energetic descriptors with the pyDock energetic results and structural metadata.

This generates the integrated feature table used by the downstream TCoaRse and BiModal prediction models.

**Example using `examplex/`:**

```bash
python scripts/merge_energies.py examplex/
```

**Output:**

- `<basename>_tcoarse_pydock_energies.csv`

### 6.10. ESMC embeddings

**Scripts:** `emb_generator.py`, `predictor_esmc.py`

Generates protein language model embeddings using ESMC.

`emb_generator.py` extracts normalized ESMC representations from the TCR–pMHC metadata.

The embeddings are saved in HDF5 format and subsequently used by `predictor_esmc.py` to generate ESMC-based predictions.

**Example using `examplex/`:**

```bash
python scripts/emb_generator.py examplex_metadata.csv
python scripts/predictor_esmc.py examplex_embeddings.h5
```

**Outputs:**

- `<basename>_embeddings.h5`
- `<basename>_esmc_predictions.csv`

### 6.11. TCoaRse predictions

**Script:** `predictor_tcoarse.py`

Applies the pretrained TCoaRse model (`tcoarse.json`) to the integrated energetic and structural feature table.

The model generates TCR–pMHC pairing predictions based on the TCoaRse representation.

**Example using `examplex/`:**

```bash
python scripts/predictor_tcoarse.py examplex_tcoarse_pydock_energies.csv
```

**Output:**

- `<basename>_tcoarse_predictions.csv`

### 6.12. BiModal predictions

**Script:** `predictor_bimodal.py`

Applies the pretrained BiModal model (`bimodal.json`) to the integrated energetic features and ESMC embeddings.

The model combines structural/energetic information with protein language representations to generate TCR–pMHC pairing predictions.

**Example using `examplex/`:**

```bash
python scripts/predictor_bimodal.py examplex_tcoarse_pydock_energies.csv examplex_embeddings.h5
```

**Output:**

- `<basename>_bimodal_predictions.csv`

---

## 7. Final output files

The downstream analysis generates the following main results:

| Output | Description |
|---|---|
| `<basename>_metrics.csv` | AF3 structural quality metrics and quality classification |
| `<basename>_metadata.csv` | Metadata extracted from predicted PDB structures |
| `<basename>_pdb/` | Processed AF3 PDB structures |
| `<basename>_cm/` | Residue-level interface contact maps |
| `sim_seq.csv` | Sequence similarity to the reference dataset |
| `sim_str.csv` | Structural similarity to the reference dataset |
| `<basename>_pairwise_dockq.csv` | Pairwise structural similarity between predicted models |
| `<basename>_pydock_ene.tar` | pyDock energetic results |
| `<basename>_tcoarse_energies.csv` | TCoaRse energetic descriptors |
| `<basename>_tcoarse_pydock_energies.csv` | Integrated TCoaRse and pyDock energetic descriptors |
| `<basename>_embeddings.h5` | Normalized ESMC embeddings |
| `<basename>_tcoarse_predictions.csv` | TCoaRse prediction scores |
| `<basename>_bimodal_predictions.csv` | BiModal prediction scores |
| `<basename>_esmc_predictions.csv` | ESMC prediction scores |

## 8. Execution

The downstream analysis is performed by executing the individual Python scripts according to their dependencies.

The general execution order is:

1. Calculate structural quality metrics and classify structures.
2. Extract and organize AF3 PDB structures.
3. Generate structural metadata.
4. Calculate sequence and structural similarities.
5. Generate interface contact maps.
6. Calculate pyDock energetic descriptors.
7. Calculate TCoaRse energetic descriptors.
8. Merge energetic descriptors and metadata.
9. Generate ESMC embeddings.
10. Generate TCoaRse, BiModal and ESMC predictions.

The scripts support parallel execution where implemented, with the number of workers configurable through their respective command-line arguments. The examples in Sections 6.2–6.12 show how the `examplex/` input is passed through the individual stages. Execute each stage after its required input files have been generated.

## 9. Repository structure

A simplified representation of the repository is:

```text
TCoaRse/
│
├── scripts/
│   ├── process_folder.py
│   ├── quality_tier.py
│   ├── cp_models.py
│   ├── metadata_from_str.py
│   ├── similarities_af3.py
│   ├── contact_maps.py
│   ├── pw_sim.py
│   ├── energetic_scorer.py
│   ├── merge_energies.py
│   ├── emb_generator.py
│   ├── predictor_tcoarse.py
│   ├── predictor_bimodal.py
│   └── predictor_esmc.py
│
├── pretrained_models/
│   ├── potential/
│   ├── rf_quality.pkl
│   ├── tcoarse.json
│   ├── bimodal.json
│   └── esmc.json
│
├── data/
│   ├── af3_training.csv
│   └── pdbs/
│
├── src/
│   ├── pdockq.py
│   ├── pdockq2_pae.py
│   └── ipsae.py
│
└── pydock/
    ├── config.yaml
    ├── 01_make_manifest.py
    ├── 02_make_chunks.py
    ├── 03_validate_chain_mapping.py
    └── worker_chunk.py
```

**Note:** The downstream analysis requires the pretrained models, reference structures, energetic potentials, external structural quality assessment scripts and pyDock Singularity container to be available at the configured paths.