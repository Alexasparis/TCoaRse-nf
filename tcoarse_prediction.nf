nextflow.enable.dsl = 2

// ---------------- PARAMETERS ----------------

params.input_dir1 = "${projectDir}/pretrained_models/potential"

params.input_file1 = "${projectDir}/pretrained_models/tcoarse.json"
params.input_file2 = "${projectDir}/pretrained_models/bimodal.json"
params.input_file3 = "${projectDir}/pretrained_models/esmc.json"

params.basename = params.af3_outputs.tokenize('/').last()

params.outdir1  = "${params.basename}_pdb"
params.outdir2  = "${params.basename}_cm"

params.outfile1 = "${params.basename}_metrics.csv"
params.outfile2 = "${params.basename}_metadata.csv"
params.outfile3 = "${params.basename}_tcoarse_energies.csv"
params.outfile4 = "${params.basename}_tcoarse_pydock_energies.csv"
params.outfile5 = "${params.basename}_tcoarse_predictions.csv"
params.outfile6 = "${params.basename}_bimodal_predictions.csv"
params.outfile7 = "${params.basename}_esmc_predictions.csv"
params.outfile8 = "${params.basename}_embeddings.h5"

params.af3_training = "${projectDir}/data/af3_training.csv"
params.af3_pdbs = "${projectDir}/data/pdbs"

params.scripts_dir = "${projectDir}/scripts"

params.cpus =16

// ---------------- QUALITY ----------------

process process_folder {

    cpus params.cpus
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path pdockq
    path pdockq2
    path ipsae

    output:
    path params.outfile1

    script:
    """
    python ${params.scripts_dir}/process_folder.py \
        ${params.af3_outputs} \
        --output ${params.outfile1} \
        --pdockq-script ${pdockq} \
        --pdockq2-script ${pdockq2} \
        --ipsae-script ${ipsae} \
        --workers ${task.cpus}
    """
}

process quality_tier {

    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path metrics_csv

    output:
    path params.outfile1

    script:
    """
    python ${params.scripts_dir}/quality_tier.py \
        --test ${metrics_csv} \
        --model ${projectDir}/pretrained_models/rf_quality.pkl \
        --output ${params.outfile1}
    """
}

// ---------------- ESMC ----------------
process generate_embeddings {

    cpus params.cpus
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path metadata_csv

    output:
    path params.outfile8

    script:
    """
    python ${params.scripts_dir}/emb_generator.py \
        -i ${metadata_csv} \
        -o ${params.outfile8} \
        -norm \
        -d cuda \
        --no-compile
    """
}

process predictor_esmc {

    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path metadata_csv
    path embeddings_h5

    output:
    path params.outfile7

    script:
    """
    python ${params.scripts_dir}/predictor_esmc.py \
        -df ${metadata_csv} \
        -emb ${embeddings_h5} \
        -m ${params.input_file3} \
        -out ${params.outfile7}
    """
}

// ---------------- PDBS ----------------

process cp_models {

    cpus params.cpus
    publishDir "${params.results_dir}", mode: 'copy'

    output:
    path params.outdir1

    script:
    """
    python ${params.scripts_dir}/cp_models.py \
        ${params.af3_outputs} \
        ${params.outdir1} \
        --workers ${task.cpus}
    """
}

process metadata_from_str {

    cpus params.cpus
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path pdb_dir

    output:
    path params.outfile2

    script:
    """
    python ${params.scripts_dir}/metadata_from_str.py \
        ${pdb_dir} \
        -o ${params.outfile2}
    """
}

// ---------------- SIMILARITIES ----------------
process similarities_af3 {

    cpus params.cpus
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path metadata_csv
    path pdb_dir

    output:
    path "sim_seq.csv"
    path "sim_str.csv"

    script:
    """
    python ${params.scripts_dir}/similarities_af3.py \
        -pre ${params.af3_training} \
        -post ${metadata_csv} \
        -pre_pdb ${params.af3_pdbs} \
        -post_pdb ${pdb_dir} \
        -o_seq sim_seq.csv \
        -o_str sim_str.csv 
    """
}

// ---------------- PYDOCK ----------------

process pydock {

    cpus params.cpus
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path pdb_dir

    output:
    path "${params.basename}_pydock_ene.tar", emit: pydock_tar

    script:
    """
    cp ${projectDir}/pydock/config.yaml ./config.yaml

    sed -i "s|__INPUT_DIR__|${pdb_dir}|g" config.yaml
    sed -i "s|__OUTPUT_DIR__|pydock_output|g" config.yaml

    python3 ${projectDir}/pydock/01_make_manifest.py --config config.yaml
    python3 ${projectDir}/pydock/02_make_chunks.py --config config.yaml
    python3 ${projectDir}/pydock/03_validate_chain_mapping.py --config config.yaml

    python3 ${projectDir}/pydock/worker_chunk.py \
        --config config.yaml \
        --chunk pydock_output/chunks/chunk_000000.tsv \
        --chunk-id 0 \
        --local-sif /gpfs/projects/hpcsharing/space/376000/pydock/singularity_container/pydock3_cythonize_20260622.sif \
        --cpus ${task.cpus}

    cp pydock_output/archives/shards/*/chunk_000000.tar ${params.basename}_pydock_ene.tar
    """
}

// ---------------- CONTACT MAPS ----------------

process contact_maps {

    cpus params.cpus
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path pdb_dir

    output:
    path params.outdir2

    script:
    """
    python ${params.scripts_dir}/contact_maps.py \
        -pdb ${pdb_dir} \
        -out ${params.outdir2} \
        -notexp \
        -workers ${task.cpus}
    """
}

process pairwise_dockq {

    cpus params.cpus
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path pdb_dir

    output:
    path "${params.basename}_pairwise_dockq.csv"

    script:
    """
    python ${params.scripts_dir}/pw_sim.py \
        --folder ${pdb_dir} \
        --output ${params.basename}_pairwise_dockq.csv \
        --workers ${task.cpus}
    """
}

// ---------------- ENERGETIC SCORER ----------------

process energetic_scorer {

    cpus params.cpus
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path cm_dir
    path pdb_dir

    output:
    path params.outfile3

    script:
    """
    python ${params.scripts_dir}/energetic_scorer.py \
        -pdb ${pdb_dir} \
        -cmd ${cm_dir} \
        -pot ${params.input_dir1} \
        -out ${params.outfile3} \
        -notexp \
        -w ${task.cpus}
    """
}

// ---------------- MERGE ----------------

process merge_energies {

    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path energies_csv
    path metadata_csv
    path pydock_tar

    output:
    path params.outfile4

    script:
    """
    python ${params.scripts_dir}/merge_energies.py \
        -tcoarse ${energies_csv} \
        -metadata ${metadata_csv} \
        -tar ${pydock_tar} \
        -o ${params.outfile4}
    """
}

// ---------------- PREDICTOR ----------------

process predictor_tcoarse {

    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path merged_csv

    output:
    path params.outfile5

    script:
    """
    python ${params.scripts_dir}/predictor_tcoarse.py \
        -df ${merged_csv} \
        -m ${params.input_file1} \
        -out ${params.outfile5}
    """
}

process predictor_bimodal {

    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path merged_csv
    path embeddings_h5

    output:
    path params.outfile6

    script:
    """
    python ${params.scripts_dir}/predictor_bimodal.py \
        -df ${merged_csv} \
        -emb ${embeddings_h5} \
        -m ${params.input_file2} \
        -out ${params.outfile6}
    """
}

// ---------------- WORKFLOW ----------------

workflow {

    pdockq_ch  = Channel.fromPath("${projectDir}/src/pdockq.py")
    pdockq2_ch = Channel.fromPath("${projectDir}/src/pdockq2_pae.py")
    ipsae_ch   = Channel.fromPath("${projectDir}/src/ipsae.py")

    process_folder(pdockq_ch, pdockq2_ch, ipsae_ch)
    quality_tier(process_folder.out)

    cp_models()
    metadata_from_str(cp_models.out)
    similarities_af3(metadata_from_str.out, cp_models.out)
    
    generate_embeddings(metadata_from_str.out)
    predictor_esmc(metadata_from_str.out, generate_embeddings.out)

    pydock(cp_models.out)
    contact_maps(cp_models.out)
    pairwise_dockq(cp_models.out)

    energetic_scorer(contact_maps.out, cp_models.out)
    merge_energies(energetic_scorer.out,metadata_from_str.out,pydock.out.pydock_tar)

    predictor_tcoarse(merge_energies.out)
    predictor_bimodal(merge_energies.out, generate_embeddings.out)
}
