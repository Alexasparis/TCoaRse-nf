# New scripts: 
```text
stitch.py 
pmhc_filtering.py
af3_jsons.py
```

1. Prepares TCR–pMHC input data.
2. Evaluates peptide–MHC binding, processing, and cleavage using multiple predictors.
3. Filters candidate pMHCs with predefined or custom filters.
4. Generates AlphaFold3 (AF3) input JSON files for TCR–pMHC structure prediction.

## Pipeline overview

```text
Input CSV
   ▼
stitch.py
   ├── Prepare TCR–pMHC data
   ├── Generate TCR sequences
   ├── Add MHC sequences
   └── Run pMHC predictors
   ▼
pmhc_filtering.py
   ├── Merge prediction results
   └── Filter candidate pMHCs
   ▼
af3_jsons.py
   └── Generate AF3 JSON files
   ▼
AlphaFold3
   └── TCR–pMHC structure prediction
   ▼
Downstream pipeline
```

# 1. Input CSV

The input must contain:

* **TCR:** either `TRAV, TRAJ, TRBV, TRBJ, CDR3a, CDR3b` or `TCRA, TCRB`
* **MHC:** either `MHC_allele` or `MHC_seq`
* **Epitope:** `Epitope`
* `tcr_id` is optional and automatically generated if missing.

# 2. Prepare TCR–pMHC data

```bash
python stitch.py \
    --in_df ../input.csv \
    --out_f ../out_f
```

`stitch.py` reconstructs TCR sequences with **Stitchr/Thimble** when needed, adds MHC sequences from `data/hla_prot.fasta`, and runs:

* NetMHCpan
* MHCflurry
* PredIG

Output: `tcrpmhc_full.csv`

# 3. Filter pMHCs

```bash
python pmhc_filtering.py \
    --out_f ../out_f \
    --tcrpmhcs ../out_f/tcrpmhc_full.csv
```

Prediction results are merged and candidate pMHCs are filtered using predefined or custom filters (format: NetCleave_strong + BA_rank_strong + NOAH_weak).

Outputs:
* `pmhc_scores.csv`
* `tcrpmhc_filtered.csv`

# 4. Generate AlphaFold3 inputs

```bash
python af3_jsons.py \
    --in_df ../out_f/tcrpmhc_filtered.csv \
    --out ../out_f/af3_jsons
```

Each JSON contains the five chains required for a TCR–pMHC class I complex:

| Chain | Component        |
| ----- | ---------------- |
| A     | MHC-I            |
| B     | β2-microglobulin |
| C     | Peptide          |
| D     | TCR α            |
| E     | TCR β            |

# 5. AlphaFold3 and downstream analysis

The generated JSON files are then run with **AlphaFold3**. 
The predicted structures are subsequently passed to the downstream structural pipeline for confidence assessment, structural filtering, interface analysis, energy calculations, structural descriptors, scoring, and TCR–pMHC ranking.
