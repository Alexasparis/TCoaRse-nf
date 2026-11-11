#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Place this file in the TCoaRse repository root and activate its Python environment.
// AF3: nextflow run downstream.nf --af3_dir examples --quality --similarity --consensus -resume
// Intermediate outputs are copied to --results_dir (default: results).
// PDB: nextflow run downstream.nf --pdb_dir results/examples_pdb --device cpu --workers 8 -resume
// Reuse: nextflow run downstream.nf --pdb_dir results/examples_pdb --pydock_archive pydock_ene.tar --embeddings_h5 embeddings.h5 -resume

params.af3_dir = null
params.pdb_dir = null
params.pydock_archive = null
params.embeddings_h5 = null
params.results_dir = 'results'
params.outdir = 'final_results'
params.workers = 10
params.chain_map = 'D:E:C:B:A'
params.device = 'cuda'
params.batch_size = null
params.normalized = false
params.quality = false
params.similarity = false
params.consensus = false
params.pre_csv = null
params.pre_pdb = null

def quote(value) {
    return "'" + value.toString().replace("'", "'\"'\"'") + "'"
}

process COLLECT_PDBS {
    publishDir params.results_dir, mode: 'copy'
    cpus { params.workers as int }

    input:
    path af3

    output:
    path 'models'

    script:
    """
    python ${quote(projectDir + "/scripts/cp_models.py")} "${af3}" models --workers ${task.cpus}
    """
}

process METADATA {
    publishDir params.results_dir, mode: 'copy'

    input:
    path pdbs

    output:
    path 'metadata.csv'

    script:
    """
    python ${quote(projectDir + "/scripts/metadata_from_str.py")} "${pdbs}" -o metadata.csv
    """
}

process CONFIDENCE {
    publishDir "${params.results_dir}/quality", mode: 'copy'
    cpus { params.workers as int }

    input:
    path af3

    output:
    path 'confidence.csv'

    script:
    """
    python ${quote(projectDir + "/scripts/process_folder.py")} "${af3}" --output confidence.csv --workers ${task.cpus}
    """
}

process QUALITY_TIERS {
    publishDir "${params.results_dir}/quality", mode: 'copy'

    input:
    path confidence

    output:
    path 'quality.csv'

    script:
    """
    python ${quote(projectDir + "/scripts/quality_tier.py")} --test "${confidence}" --output quality.csv
    """
}

process AF3_SIMILARITY {
    publishDir "${params.results_dir}/af3_similarity", mode: 'copy'
    cpus { params.workers as int }

    input:
    path pdbs, name: 'query_pdbs'
    path metadata

    output:
    path 'sim_seq.csv', emit: seq
    path 'sim_str.csv', emit: str
    path 'query_pdbs_merged', optional: true

    script:
    def preCsv = params.pre_csv ? "-pre ${quote(file(params.pre_csv).toAbsolutePath())}" : ''
    def prePdb = params.pre_pdb ? "-pre_pdb ${quote(file(params.pre_pdb).toAbsolutePath())}" : ''
    """
    python ${quote(projectDir + "/scripts/similarities_af3.py")} -pdb_info "${metadata}" -pdb_dir "${pdbs}" \\
      -out_seq sim_seq.csv -out_str sim_str.csv -workers ${task.cpus} ${preCsv} ${prePdb}
    """
}

process REPLICA_CONSENSUS {
    publishDir "${params.results_dir}/pairwise", mode: 'copy'
    cpus { params.workers as int }

    input:
    path pdbs

    output:
    path 'consensus.csv'

    script:
    """
    python ${quote(projectDir + "/scripts/pw_sim.py")} --folder "${pdbs}" --output consensus.csv --workers ${task.cpus}
    """
}

process CONTACT_MAPS {
    publishDir params.results_dir, mode: 'copy'
    cpus { params.workers as int }

    input:
    path pdbs

    output:
    path 'contacts'

    script:
    """
    python ${quote(projectDir + "/scripts/contact_maps.py")} -pdb "${pdbs}" -out contacts -workers ${task.cpus} -cm ${quote(params.chain_map)}
    """
}

process PYDOCK {
    publishDir params.results_dir, mode: 'copy', pattern: 'pydock_results'
    cpus { params.workers as int }

    input:
    path pdbs

    output:
    path 'pydock.tar', emit: archive
    path 'pydock_results'

    script:
    """
    python ${quote(projectDir + "/scripts/run_pydock.py")} --input "${pdbs}" --output pydock_results/
    python - <<'PY'
    from pathlib import Path
    from shutil import copyfile

    archives = sorted(Path("pydock_results").rglob("*_ene.tar"))
    if len(archives) != 1:
        raise SystemExit(f"Expected one *_ene.tar archive, found {len(archives)}: {archives}")
    copyfile(archives[0], "pydock.tar")
    PY
    """
}

process TCOARSE_ENERGIES {
    publishDir "${params.results_dir}/energies", mode: 'copy'
    cpus { params.workers as int }

    input:
    path pdbs
    path contacts
    path potential

    output:
    path 'tcoarse_energies.csv'

    script:
    """
    python ${quote(projectDir + "/scripts/energetic_scorer.py")} -cmd "${contacts}" -pdb "${pdbs}" -pot "${potential}" \\
      -chains ${quote(params.chain_map)} -out tcoarse_energies.csv -t 7 -w ${task.cpus}
    """
}

process MERGE_ENERGIES {
    publishDir "${params.results_dir}/energies", mode: 'copy'

    input:
    path energies
    path archive, name: 'input_pydock.tar'
    path metadata

    output:
    path 'merged_energies.csv'

    script:
    """
    python ${quote(projectDir + "/scripts/merge_energies.py")} -tcoarse "${energies}" -pydock "${archive}" \\
      -metadata "${metadata}" -out merged_energies.csv
    """
}

process EMBEDDINGS {
    publishDir params.results_dir, mode: 'copy'
    label 'embeddings'
    cpus { params.device == 'cpu' ? (params.workers as int) : 1 }

    input:
    path metadata

    output:
    path 'embeddings.h5'

    script:
    def batch = params.batch_size == null ? '' : "-b ${params.batch_size as int}"
    def normalized = params.normalized ? '--normalized' : ''
    """
    export OMP_NUM_THREADS=${task.cpus}
        python ${quote(projectDir + "/scripts/emb_generator.py")} -i "${metadata}" -o embeddings.h5 \\
      -d ${quote(params.device)} --no-compile ${batch} ${normalized}
    """
}

process PREDICT_STR {
    publishDir "${params.results_dir}/predictions", mode: 'copy'

    input:
    path energies
    path model

    output:
    path 'str_predictions.csv'

    script:
    """
    python ${quote(projectDir + "/scripts/predictor_tcoarse_str.py")} -df "${energies}" \\
      -m "${model}" -out str_predictions.csv
    """
}

process PREDICT_SEQ {
    publishDir "${params.results_dir}/predictions", mode: 'copy'

    input:
    path metadata
    path embeddings, name: 'input_embeddings.h5'
    path model

    output:
    path 'seq_predictions.csv'

    script:
    """
    python ${quote(projectDir + "/scripts/predictor_tcoarse_seq.py")} -df "${metadata}" -emb "${embeddings}" \\
      -m "${model}" -out seq_predictions.csv
    """
}

process PREDICT_BI {
    publishDir "${params.results_dir}/predictions", mode: 'copy'

    input:
    path energies
    path embeddings, name: 'input_embeddings.h5'
    path model

    output:
    path 'bi_predictions.csv'

    script:
    """
    python ${quote(projectDir + "/scripts/predictor_tcoarse_bi.py")} -df "${energies}" -emb "${embeddings}" \\
      -m "${model}" -out bi_predictions.csv
    """
}

// Run the existing exporter with temporary, header-only inputs for skipped
// analyses. Remove their exports and retain IDs from actual similarity tables.
process FINAL_OUTPUT {
    publishDir params.outdir, mode: 'copy', saveAs: { name -> name.tokenize('/').last() }

    input:
    path metadata
    path energies
    path str_predictions
    path seq_predictions
    path bi_predictions
    path quality
    path sim_seq
    path sim_str
    path consensus

    output:
    path 'final/*.csv'

    script:
    def qualityArg = params.quality ? quality : 'empty_quality.csv'
    def seqArg = params.similarity ? sim_seq : 'empty_sim_seq.csv'
    def strArg = params.similarity ? sim_str : 'empty_sim_str.csv'
    def consensusArg = params.consensus ? consensus : 'empty_consensus.csv'
    """
    printf 'tcr_id,model_number\\n' > empty_quality.csv
    printf 'tcr_id\\n' > empty_sim_seq.csv
    printf 'tcr_id,model_number\\n' > empty_sim_str.csv
    printf 'tcr_id\\n' > empty_consensus.csv

    python ${quote(projectDir + "/scripts/generate_final_output.py")} \\
      -metadata "${metadata}" -tcoarse_str "${str_predictions}" \\
      -tcoarse_seq "${seq_predictions}" -tcoarse_bi "${bi_predictions}" \\
      -pydock "${energies}" -pw "${consensusArg}" -qm "${qualityArg}" \\
      -sim_seq "${seqArg}" -sim_str "${strArg}" -o final

    python - <<'PY'
    from pathlib import Path
    import pandas as pd

    final = Path("final")
    exports = [
        ("${qualityArg}", "model_quality_metrics.csv", []),
        ("${consensusArg}", "replica_consensus_analysis.csv", []),
        ("${seqArg}", "af3_similarity_analysis_seq.csv", ["tcr_id"]),
        ("${strArg}", "af3_similarity_analysis_str.csv", ["tcr_id", "model_number"]),
    ]
    for source, name, keys in exports:
        output = final / name
        if source.startswith("empty_"):
            output.unlink(missing_ok=True)
        elif keys:
            original = pd.read_csv(source, dtype={"tcr_id": str})
            if any(key not in original for key in keys):
                raise SystemExit(f"Missing identifier columns in {source}: {keys}")
            if output.read_text().strip():
                filtered = pd.read_csv(output, dtype={"tcr_id": str})
                if len(filtered) != len(original):
                    raise SystemExit(f"Unexpected row count in {output}")
            else:
                filtered = pd.DataFrame(index=original.index)
            filtered = filtered.drop(columns=keys, errors="ignore")
            pd.concat([original[keys], filtered], axis=1).to_csv(output, index=False)
    PY
    """
}

workflow {
    if ((!params.af3_dir && !params.pdb_dir) || (params.af3_dir && params.pdb_dir)) {
        error 'Provide exactly one input: --af3_dir or --pdb_dir.'
    }
    if (params.quality && !params.af3_dir) {
        error '--quality requires --af3_dir with the original AF3 confidence files.'
    }

    if (params.af3_dir) {
        af3 = Channel.value(file(params.af3_dir, checkIfExists: true, type: 'dir'))
        pdbs = COLLECT_PDBS(af3)
    } else {
        pdbs = Channel.value(file(params.pdb_dir, checkIfExists: true, type: 'dir'))
    }

    metadata = METADATA(pdbs)
    contacts = CONTACT_MAPS(pdbs)
    if (params.pydock_archive) {
        archive = Channel.value(file(params.pydock_archive, checkIfExists: true, glob: false))
    } else {
        archive = PYDOCK(pdbs).archive
    }
    potential = file("${projectDir}/pretrained_models/potentials/pot_7", checkIfExists: true)
    energies = TCOARSE_ENERGIES(pdbs, contacts, potential)
    merged = MERGE_ENERGIES(energies, archive, metadata)
    if (params.embeddings_h5) {
        embeddings = Channel.value(file(params.embeddings_h5, checkIfExists: true, glob: false))
    } else {
        embeddings = EMBEDDINGS(metadata)
    }

    strModel = file("${projectDir}/pretrained_models/tcoarse_str.json", checkIfExists: true)
    seqModel = file("${projectDir}/pretrained_models/tcoarse_seq.json", checkIfExists: true)
    biModel = file("${projectDir}/pretrained_models/tcoarse_bi.json", checkIfExists: true)
    strPred = PREDICT_STR(merged, strModel)
    seqPred = PREDICT_SEQ(metadata, embeddings, seqModel)
    biPred = PREDICT_BI(merged, embeddings, biModel)

    quality = Channel.value([])
    seqSim = Channel.value([])
    strSim = Channel.value([])
    consensus = Channel.value([])
    if (params.quality) {
        quality = QUALITY_TIERS(CONFIDENCE(af3))
    }
    if (params.similarity) {
        similarities = AF3_SIMILARITY(pdbs, metadata)
        seqSim = similarities.seq
        strSim = similarities.str
    }
    if (params.consensus) {
        consensus = REPLICA_CONSENSUS(pdbs)
    }

    FINAL_OUTPUT(metadata, merged, strPred, seqPred, biPred, quality, seqSim, strSim, consensus)
}
