#!/usr/bin/env python3
import os
from anarci import anarci
from Bio.PDB import MMCIFParser, MMCIFIO, PDBIO, is_aa, PDBParser, PPBuilder
from Bio import PDB
import numpy as np
import json
import subprocess
from statistics import mean
import re
import csv
import argparse
import pandas as pd
import glob
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

residue_mapping = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

ANARCI_LINE_PATTERN = re.compile(r'^([A-Z])\s+(\d+)\s+([A-Z\-])', re.MULTILINE)

def print_verbose(message, verbose):
    if verbose:
        print(message)

def get_structure(model_file):
    if model_file.endswith(".pdb"):
        parser = PDB.PDBParser(QUIET=True)
    else:
        parser = PDB.MMCIFParser(QUIET=True)
    return parser.get_structure("structure", model_file)


def remove_low_plddt_and_get_absolute_indices(cif_file, output_file, threshold=50):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)
    chain_lengths = {chain.id: len(list(chain.get_residues())) for chain in structure[0]}
    removed_indices = {chain.id: {"N": [], "C": []} for chain in structure[0]}

    model = structure[0]
    for chain in model:
        if chain.id in ("C", "D", "E"):
            continue
        residues = list(chain.get_residues())
        chain_id = chain.id

        while residues:
            first_residue = residues[0]
            plddts = [atom.bfactor for atom in first_residue.get_atoms()]
            if any(plddt < threshold for plddt in plddts):
                removed_indices[chain_id]["N"].append(first_residue.id[1])
                chain.detach_child(first_residue.id)
                residues.pop(0)
            else:
                break

        while residues:
            last_residue = residues[-1]
            plddts = [atom.bfactor for atom in last_residue.get_atoms()]
            if any(plddt < threshold for plddt in plddts):
                removed_indices[chain_id]["C"].append(last_residue.id[1])
                chain.detach_child(last_residue.id)
                residues.pop()
            else:
                break

    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_file)

    absolute_indices = []
    global_residue_number = 1

    for chain, residues in removed_indices.items():
        chain_length = chain_lengths.get(chain, 0)
        n_terminal_residues = [r for r in residues.get("N", []) if r <= chain_length]
        c_terminal_residues = [r for r in residues.get("C", []) if r <= chain_length]
        absolute_indices.extend([global_residue_number + r - 1 for r in sorted(n_terminal_residues)])
        absolute_indices.extend([global_residue_number + r - 1 for r in sorted(c_terminal_residues)])
        global_residue_number += chain_length

    return absolute_indices, structure


def remove_atoms_from_pae_matrix(json_file, removed_atom_numbers, output_file):
    with open(json_file, 'r') as f:
        pae_data = json.load(f)
    pae_matrix = np.array(pae_data['pae'])
    if removed_atom_numbers:
        pae_matrix = np.delete(pae_matrix, removed_atom_numbers, axis=0)
        pae_matrix = np.delete(pae_matrix, removed_atom_numbers, axis=1)
    np.save(output_file, pae_matrix)
    return pae_matrix


def remove_atoms_from_pde_matrix(json_file, removed_atom_numbers, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    pde_matrix = np.array(data['contact_probs'])
    if removed_atom_numbers:
        pde_matrix = np.delete(pde_matrix, removed_atom_numbers, axis=0)
        pde_matrix = np.delete(pde_matrix, removed_atom_numbers, axis=1)
    np.save(output_file, pde_matrix)
    return pde_matrix


def cif_to_pdb(cif_file, verbose=False, structure=None):
    pdb_file = os.path.splitext(cif_file)[0] + ".pdb"
    if structure is None:
        parser = MMCIFParser(QUIET=True)
        try:
            structure = parser.get_structure(os.path.basename(pdb_file), cif_file)
        except Exception as e:
            print(f"Error parsing CIF file {cif_file}: {e}")
            return None
    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(pdb_file)
        print_verbose(f"Successfully converted {cif_file} to {pdb_file}", verbose)
    except Exception as e:
        print(f"Error writing PDB file {pdb_file}: {e}")
    return structure


def merge_pdb(pdb_file, chain_mapping={"mhc_chain": "A", "b2_chain": "B", "peptide_chain": "C",
                                        "tcra_chain": "D", "tcrb_chain": "E"}, verbose=False):
    tcra_id = chain_mapping["tcra_chain"]
    tcrb_id = chain_mapping["tcrb_chain"]
    mhc_id = chain_mapping["mhc_chain"]
    b2_id = chain_mapping["b2_chain"]
    epitope_id = chain_mapping["peptide_chain"]

    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
    dir_name = os.path.dirname(pdb_file)
    output_file_path = os.path.join(dir_name, f"{base_name}_merged.pdb")

    cleaned_pdb_file = os.path.join(dir_name, f"{base_name}_cleaned.pdb")
    cleaned_lines = remove_headers(pdb_file)
    with open(cleaned_pdb_file, 'w') as cleaned_file:
        cleaned_file.writelines(cleaned_lines)

    command_AB = (f"pdb_selchain -{tcra_id},{tcrb_id} {cleaned_pdb_file} "
                  f"| pdb_chain -B | pdb_reres -1 | pdb_delhetatm > {dir_name}/B.pdb")
    command_MB = (f"pdb_selchain -{mhc_id},{b2_id},{epitope_id} {cleaned_pdb_file} "
                  f"| pdb_chain -A | pdb_reres -1 | pdb_delhetatm > {dir_name}/A.pdb")

    try:
        subprocess.run(command_MB, shell=True, check=True)
        subprocess.run(command_AB, shell=True, check=True)

        A_lines = remove_headers(os.path.join(dir_name, "A.pdb"))
        B_lines = remove_headers(os.path.join(dir_name, "B.pdb"))

        with open(output_file_path, 'w') as outfile:
            outfile.writelines(A_lines)
            outfile.writelines(B_lines)

        os.remove(os.path.join(dir_name, 'A.pdb'))
        os.remove(os.path.join(dir_name, 'B.pdb'))
        os.remove(cleaned_pdb_file)
        print_verbose(f"Successfully merged chains {output_file_path}", verbose)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {pdb_file}: {e}")


def remove_headers(file_path):
    cleaned_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if (line.startswith("ATOM") or line.startswith("HETATM")) and len(line) > 21:
                cleaned_lines.append(line)
    return cleaned_lines


def calculate_global_plddt(cif_file_path, structure=None):
    try:
        if structure is None:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("structure", cif_file_path)
        b_factors = [
            atom.get_bfactor() for model in structure
            for chain in model
            for residue in chain
            for atom in residue]
        if not b_factors:
            print("No B-factors found in the CIF file.")
            return None
        return mean(b_factors)
    except FileNotFoundError:
        print(f"Error: The file {cif_file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error processing the CIF file: {e}")
        return None


def extract_b_factors(cdr_atoms, chain):
    b_factors = []
    for atomname, resid, resname, chainid in cdr_atoms:
        if chainid == chain.id:
            try:
                residue = chain[resid]
                if atomname in residue:
                    atom = residue[atomname]
                    b_factors.append(atom.get_bfactor())
                else:
                    print(f"Atom {atomname} not found in residue {resid} ({resname}) of chain {chain.id}")
            except KeyError:
                print(f"Residue {resid} ({resname}) not found in chain {chain.id}")
    return b_factors


def extract_sequences_from_structure(structure):
    sequences_str = {}
    sequences_tuples = {}
    for model in structure:
        for chain in model.get_chains():
            chain_id = chain.get_id()
            sequence_str = []
            sequence_tuples = []
            for residue in chain:
                if PDB.is_aa(residue):
                    res_name = residue.get_resname()
                    resid = residue.get_id()[1]
                    sequence_str.append(residue_mapping.get(res_name, 'X'))
                    sequence_tuples.append((res_name, resid))
            sequences_str[chain_id] = ''.join(sequence_str)
            sequences_tuples[chain_id] = sequence_tuples
    return sequences_str, sequences_tuples


def extract_sequences(pdb_file):
    """Kept for backward compatibility / standalone use."""
    return extract_sequences_from_structure(get_structure(pdb_file))


@lru_cache(maxsize=None)
def get_cached_anarci_mapping(sequence):
    raw_output = run_anarci(sequence)
    return tuple(parse_anarci_output(raw_output))


def compute_plddt_mean(plddt, indices, eps=1e-6):
    """Compute mean PLDDT for a set of indices."""
    indices = np.concatenate(indices)
    return ((plddt[indices].sum() + eps) / len(indices)) * 0.01


def get_chain_ca_array(chain):
    return np.array([
        res["CA"].get_bfactor() if "CA" in res else np.nan
        for res in chain])


def get_cdr_indices(sequence: str, cdr1, cdr2, cdr3):
    def cdr_to_string(cdr):
        return "".join([x[2] for x in cdr if x[2] != "-" and x[2] is not None])
    indices = []
    for cdr in [cdr1, cdr2, cdr3]:
        cdr_str = cdr_to_string(cdr)
        start_idx = sequence.find(cdr_str)
        if start_idx == -1:
            indices.append(np.array([], dtype=int))
            continue
        end_idx = start_idx + len(cdr_str)
        indices.append(np.arange(start_idx, end_idx))
    return indices


def compute_cdr_metrics(model_file, alpha_chain, beta_chain, structure=None):

    if structure is None:
        structure = get_structure(model_file)

    sequences_loose, _ = extract_sequences_from_structure(structure)
    seq_A_loose = sequences_loose[alpha_chain]
    seq_B_loose = sequences_loose[beta_chain]

    residues_A = extract_residues_and_resids_from_structure(structure, alpha_chain)
    residues_B = extract_residues_and_resids_from_structure(structure, beta_chain)

    chain_A = structure[0][alpha_chain]
    chain_B = structure[0][beta_chain]
    chain_P = structure[0]["C"]

    seq_A_std = "".join(residue_mapping.get(res.get_resname(), "X")
                         for res in chain_A if PDB.is_aa(res, standard=True))
    seq_B_std = "".join(residue_mapping.get(res.get_resname(), "X")
                         for res in chain_B if PDB.is_aa(res, standard=True))

    map_A_loose = map_imgt_to_original(list(get_cached_anarci_mapping(seq_A_loose)), residues_A)
    map_B_loose = map_imgt_to_original(list(get_cached_anarci_mapping(seq_B_loose)), residues_B)
    map_A_std = map_imgt_to_original(list(get_cached_anarci_mapping(seq_A_std)), residues_A)
    map_B_std = map_imgt_to_original(list(get_cached_anarci_mapping(seq_B_std)), residues_B)

    cdr1_A_l, cdr2_A_l, cdr3_A_l = parse_CDR1(map_A_loose), parse_CDR2(map_A_loose), parse_CDR3(map_A_loose)
    cdr1_B_l, cdr2_B_l, cdr3_B_l = parse_CDR1(map_B_loose), parse_CDR2(map_B_loose), parse_CDR3(map_B_loose)
    cdr1_A_s, cdr2_A_s, cdr3_A_s = parse_CDR1(map_A_std), parse_CDR2(map_A_std), parse_CDR3(map_A_std)
    cdr1_B_s, cdr2_B_s, cdr3_B_s = parse_CDR1(map_B_std), parse_CDR2(map_B_std), parse_CDR3(map_B_std)

    # ---- per-CDR mean pLDDT (old cdr_plddts) ----
    cdr1_atoms_A = extract_atoms_for_cdr(cdr1_A_l, model_file, alpha_chain, structure=structure)
    cdr2_atoms_A = extract_atoms_for_cdr(cdr2_A_l, model_file, alpha_chain, structure=structure)
    cdr3_atoms_A = extract_atoms_for_cdr(cdr3_A_l, model_file, alpha_chain, structure=structure)
    cdr1_atoms_B = extract_atoms_for_cdr(cdr1_B_l, model_file, beta_chain, structure=structure)
    cdr2_atoms_B = extract_atoms_for_cdr(cdr2_B_l, model_file, beta_chain, structure=structure)
    cdr3_atoms_B = extract_atoms_for_cdr(cdr3_B_l, model_file, beta_chain, structure=structure)

    def safe_mean(vals, label):
        if len(vals) == 0:
            print(f"ERROR: {label} empty in model {model_file}")
            return np.nan
        return np.mean(vals)

    mean_cdr1_A = safe_mean(extract_b_factors(cdr1_atoms_A, chain_A), "CDR1_A")
    mean_cdr2_A = safe_mean(extract_b_factors(cdr2_atoms_A, chain_A), "CDR2_A")
    mean_cdr3_A = safe_mean(extract_b_factors(cdr3_atoms_A, chain_A), "CDR3_A")
    mean_cdr1_B = safe_mean(extract_b_factors(cdr1_atoms_B, chain_B), "CDR1_B")
    mean_cdr2_B = safe_mean(extract_b_factors(cdr2_atoms_B, chain_B), "CDR2_B")
    mean_cdr3_B = safe_mean(extract_b_factors(cdr3_atoms_B, chain_B), "CDR3_B")

    peptide_plddt = [atom.get_bfactor() for residue in chain_P for atom in residue]
    if not peptide_plddt:
        print(f"WARNING: No atoms found in peptide chain C for model {model_file}")
        peptide_plddt = [np.nan]
    mean_peptide_plddt = np.nanmean(peptide_plddt)

    # ---- CA-only, index-based combined score (old cdr_plddts_exact) ----
    bf_A = get_chain_ca_array(chain_A)
    bf_B = get_chain_ca_array(chain_B)
    bf_P = get_chain_ca_array(chain_P)

    plddt_A = np.concatenate([bf_A, bf_P])
    plddt_B = np.concatenate([bf_B, bf_P])
    idx_A = get_cdr_indices(seq_A_std, cdr1_A_s, cdr2_A_s, cdr3_A_s)
    idx_B = get_cdr_indices(seq_B_std, cdr1_B_s, cdr2_B_s, cdr3_B_s)
    pep_A = [np.arange(len(bf_A), len(bf_A) + len(bf_P))]
    pep_B = [np.arange(len(bf_B), len(bf_B) + len(bf_P))]
    score_A = compute_plddt_mean(plddt_A, idx_A + pep_A)
    score_B = compute_plddt_mean(plddt_B, idx_B + pep_B)
    mean_all_cdrs_peptide = (score_A + score_B) / 2

    return (mean_cdr1_A, mean_cdr2_A, mean_cdr3_A, mean_cdr1_B, mean_cdr2_B, mean_cdr3_B,
            mean_peptide_plddt, mean_all_cdrs_peptide)


def calculate_iptms(json_file_path, length=5):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        chain_iptm = data.get('chain_iptm', [])
        chain_iptm_mean = sum(chain_iptm) / len(chain_iptm) if chain_iptm else None
        if not chain_iptm:
            print("No data found in 'chain_iptm'.")

        chain_ptm = data.get('chain_ptm', [])
        chain_ptm_mean = sum(chain_ptm) / len(chain_ptm) if chain_ptm else None
        if not chain_ptm:
            print("No data found in 'chain_ptm'.")

        chain_pair_iptm = data.get('chain_pair_iptm', [])
        if not chain_pair_iptm:
            print("No data found in 'chain_pair_iptm'.")
            tcr_pmch_iptm = None
        else:
            if length == 5:
                tcr_pmch_pairs = [
                    chain_pair_iptm[0][3], chain_pair_iptm[0][4],
                    chain_pair_iptm[2][3], chain_pair_iptm[2][4]]
                tcr_pmch_iptm = sum(tcr_pmch_pairs) / len(tcr_pmch_pairs)
            elif length == 4:
                tcr_pmch_pairs = [
                    chain_pair_iptm[0][2], chain_pair_iptm[0][3],
                    chain_pair_iptm[2][2], chain_pair_iptm[2][3]]
                tcr_pmch_iptm = sum(tcr_pmch_pairs) / len(tcr_pmch_pairs)

        return chain_iptm_mean, chain_ptm_mean, tcr_pmch_iptm

    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return None
    except KeyError as e:
        print(f"Missing key in JSON data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def calculate_pdockq(model_file, pdockq_script):
    command = f"python {pdockq_script} --pdbfile {model_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    pdockq = float(result.stdout.split('=')[1].split(' ')[1])
    return result.stdout, pdockq


def calculate_pdockq2(model_file, pde_mtx, pae_mtx, pdockq2_script):
    python_path = "python"
    command = f'"{python_path}" "{pdockq2_script}" -pde "{pde_mtx}" -pae "{pae_mtx}" -pdb "{model_file}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        return None, None, None, None, None
    lines = [line for line in result.stdout.strip().split('\n') if line.strip()]

    try:
        ipde_A = float(lines[1].split()[1])
        ipde_B = float(lines[2].split()[1])
        ipae_A = float(lines[4].split()[1])
        ipae_B = float(lines[5].split()[1])
        pdockq2_A = float(lines[7].split()[1])
        pdockq2_B = float(lines[8].split()[1])
    except (IndexError, ValueError) as e:
        print("Error parsing pdockq2_pae.py:", e)
        print("Output:\n", result.stdout)
        return None, None, None, None, None
    return result.stdout, ipde_A, ipde_B, ipae_A, ipae_B, pdockq2_A, pdockq2_B


def extract_residues_and_resids_from_structure(structure, chain_id):
    residues = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    resid = residue.get_id()[1]
                    resname = residue.get_resname().upper()
                    residue_one_letter = residue_mapping.get(resname, 'X')
                    residues.append((resid, residue_one_letter))
    return residues


def extract_residues_and_resids(pdb_file, chain_id):
    """Kept for backward compatibility / standalone use. Always used MMCIFParser
    in the original code (residue/CDR bookkeeping here always operates on the
    cif file), so that behavior is preserved."""
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    return extract_residues_and_resids_from_structure(structure, chain_id)


def run_anarci(sequence):
    results, alignment_details, hit_tables = anarci([("query", sequence)], scheme="imgt")
    return results


def parse_anarci_output(results):
    imgt_numbered_seq = []
    numbering = results[0][0][0]
    for (imgt_num, insertion), residue in numbering:
        if insertion.strip():
            continue
        imgt_numbered_seq.append((imgt_num, residue))
    return imgt_numbered_seq


def map_imgt_to_original(imgt_numbered_seq, pdb_resids):
    mapping = []
    pdb_resid_index = 0

    for imgt_pos, residue in imgt_numbered_seq:
        if residue != "-":
            for original_resid, residue1 in pdb_resids[pdb_resid_index:]:
                if residue1 == residue:
                    mapping.append((original_resid, imgt_pos, residue))
                    pdb_resid_index += 1
                    break
                else:
                    pdb_resid_index += 1
            else:
                mapping.append((None, imgt_pos, residue))
        else:
            mapping.append((None, imgt_pos, residue))
    return mapping


def parse_CDR3(mapping):
    return [t for t in mapping if 104 <= t[1] <= 118 and t[2] != "-"]


def parse_CDR2(mapping):
    return [t for t in mapping if 56 <= t[1] <= 65 and t[2] != "-"]


def parse_CDR1(mapping):
    return [t for t in mapping if 27 <= t[1] <= 38 and t[2] != "-"]


def extract_atoms_for_cdr(cdr_list, pdb_file, chain_id, structure=None):
    """
    OPT: accepts an optional pre-parsed `structure` so callers that need this
    for several CDRs of the same model only pay the parsing cost once.
    Also replaced the O(n_residues * n_cdr) nested-loop lookup with an O(1)
    set lookup.
    """
    if structure is None:
        structure = get_structure(pdb_file)

    cdr_set = {(resid, resname) for resid, _imgt, resname in cdr_list}

    atom_list = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    resid = residue.get_id()[1]
                    resname_3 = residue.get_resname()
                    resname_1 = residue_mapping.get(resname_3, 'X')
                    if (resid, resname_1) in cdr_set:
                        for atom in residue:
                            atom_list.append((atom.get_name(), resid, resname_3, chain.id))
    return atom_list


def get_seq_from_pdb_chain(pdb_file, chain_id):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    ppb = PDB.PPBuilder()
    actual_seq = ""
    if chain_id in model:
        for pp in ppb.build_peptides(model[chain_id]):
            actual_seq += str(pp.get_sequence())
    return actual_seq


def get_all_chain_sequences_from_structure(structure):
    model = structure[0]
    ppb = PDB.PPBuilder()
    seqs = {}
    for chain in model:
        seq = ""
        for pp in ppb.build_peptides(chain):
            seq += str(pp.get_sequence())
        seqs[chain.id] = seq
    return seqs


def get_all_chain_sequences(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    return get_all_chain_sequences_from_structure(structure)


def get_sync_indices_for_pae(json_path, pdb_path, verbose=False, pdb_structure=None):
    """OPT: accepts an optional pre-parsed `pdb_structure` (e.g. the one
    already parsed for has_tcr_peptide_contact) to avoid re-parsing pdb_path
    from disk a second time."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    original_seqs = {}
    for item in data.get('sequences', []):
        if 'protein' in item:
            original_seqs[item['protein']['id']] = item['protein']['sequence']

    ordered_chains = sorted(original_seqs.keys())
    if pdb_structure is not None:
        pdb_seqs = get_all_chain_sequences_from_structure(pdb_structure)
    else:
        pdb_seqs = get_all_chain_sequences(pdb_path)

    absolute_removed_indices = []
    current_pae_offset = 0
    for chain_id in ordered_chains:
        full_seq = original_seqs[chain_id]
        len_full = len(full_seq)
        pdb_seq = pdb_seqs.get(chain_id, "")
        print_verbose(f"Chain {chain_id}:", verbose)
        print_verbose(f"Full length {len_full}", verbose)
        print_verbose(f"PDB length {len(pdb_seq)}", verbose)

        if not pdb_seq:
            print_verbose(f"Chain {chain_id} missing in PDB, removing all its residues from PAE.", verbose)
            for i in range(len_full):
                absolute_removed_indices.append(current_pae_offset + i)
        else:
            start_rel = full_seq.find(pdb_seq)
            if start_rel != -1:
                for i in range(0, start_rel):
                    absolute_removed_indices.append(current_pae_offset + i)
                print_verbose(f"Residues removed at start (N-term): {start_rel}", verbose)
                end_rel = start_rel + len(pdb_seq)
                for i in range(end_rel, len_full):
                    absolute_removed_indices.append(current_pae_offset + i)
                print_verbose(f"Residues removed at end (C-term): {len_full - end_rel}", verbose)
            else:
                print(f"Warning: Sequence mismatch in chain {chain_id}")

        current_pae_offset += len_full

    return absolute_removed_indices


def has_tcr_peptide_contact(structure, cutoff=10.0):
    tcr_atoms = {'D': [], 'E': []}
    pep_atoms = []

    model = structure[0]

    for chain in model:
        cid = chain.id
        if cid == 'C':
            for residue in chain:
                for atom in residue:
                    pep_atoms.append(atom.get_coord())
        if cid in ('D', 'E'):
            for residue in chain:
                for atom in residue:
                    tcr_atoms[cid].append(atom.get_coord())

    if not pep_atoms:
        return False
    pep_arr = np.array(pep_atoms)

    for cid in ('D', 'E'):
        if not tcr_atoms[cid]:
            continue
        tcr_arr = np.array(tcr_atoms[cid])
        diffs = tcr_arr[:, None, :] - pep_arr[None, :, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=2))
        if np.min(dists) <= cutoff:
            return True
    return False


def calculate_ipsae(cif_file_path, json_conf_path, scripts_dir, verbose=False):
    command = ["python", os.path.join(scripts_dir, "ipsae.py"), json_conf_path, cif_file_path, "10", "10"]
    print(f"Calculating iPSAE for {cif_file_path}...")
    print_verbose(" ".join(command), verbose)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error calculating iPSAE for {cif_file_path}: {result.stderr}")


def parse_ipsae_output(output_file_path):
    results = {}
    try:
        with open(output_file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        header = lines[0].split()

        target_cols = ["ipSAE", "ipSAE_d0chn", "ipSAE_d0dom", "ipTM_d0chn", "LIS"]
        col_idx = {col: header.index(col) for col in target_cols}

        for line in lines[1:]:
            fields = line.split()
            chn1 = fields[header.index("Chn1")]
            chn2 = fields[header.index("Chn2")]
            row_type = fields[header.index("Type")]
            key = f"{chn1}_{chn2}_{row_type}"
            results[key] = {col: float(fields[idx]) for col, idx in col_idx.items()}

        return results
    except Exception as e:
        print(f"Error parsing ipSAE output file {output_file_path}: {e}")
        return {}


def calculate_for_seed(folder_path, cif_file, confidence_file, scripts_dir):
    def get_ipsae_path():
        return next((os.path.join(folder_path, f) for f in os.listdir(folder_path)
                     if f.endswith(".txt") and not f.endswith("byres.txt")), None)

    ipsae_path = get_ipsae_path()
    if ipsae_path is None:
        print(f"iPSAE results not found in {folder_path}. Running calculation...")
        if cif_file and confidence_file:
            calculate_ipsae(cif_file, confidence_file, scripts_dir=scripts_dir, verbose=False)
            ipsae_path = get_ipsae_path()
        else:
            print(f"Error: Missing CIF or JSON file in {folder_path}")
            return None

    if ipsae_path:
        ipsae_results = parse_ipsae_output(ipsae_path)

    interfaces = ["A_D", "A_E", "C_D", "C_E"]
    results = {}

    for interface in interfaces:
        entry = ipsae_results.get(f"{interface}_max", {})
        results[interface] = (
            entry.get("ipSAE"), entry.get("ipSAE_d0chn"), entry.get("ipSAE_d0dom"),
            entry.get("ipTM_d0chn"), entry.get("LIS"))

    def avg(idx):
        vals = [results[i][idx] for i in interfaces if results[i][idx] is not None]
        return sum(vals) / len(interfaces)

    return avg(0), avg(1), avg(2), avg(3), avg(4)


def process_single_seed(seed_folder, folder_path, tcr_id, threshold, ranking_df, json_input_path,
                         pdockq_script, pdockq2_script, ipsae_scripts_dir, verbose=False):

    if not seed_folder.startswith("seed"):
        return None

    model_number = seed_folder.split("-")[-1]
    seed_number = seed_folder.split("-")[1].split("_")[0]
    print_verbose(f"\nProcessing model {model_number} in TCR {tcr_id}...\n", verbose)
    seed_folder_path = os.path.join(folder_path, seed_folder)

    model_file_path = None
    file_output_path = None
    summary_json_file_path = None
    confidence_json_file_path = None
    merged_pdb = None
    pdb_file_path = None
    contacts_ok = None
    cif_structure = None
    pdb_structure = None
    files = sorted(os.listdir(seed_folder_path))

    for file in files:
        if file.endswith("model.cif"):
            model_file_path = os.path.join(seed_folder_path, file)
            print_verbose(f"Model file path,{model_file_path}", verbose)

            file_output_path = os.path.join(seed_folder_path, "model_cleaned.cif")
            removed_atom_numbers, cif_structure = remove_low_plddt_and_get_absolute_indices(
                model_file_path, file_output_path, threshold=threshold)
            print_verbose(f"File output path: {file_output_path}", verbose)
            cif_to_pdb(file_output_path, structure=cif_structure)
            pdb_file_path = os.path.splitext(file_output_path)[0] + ".pdb"

            merge_pdb(pdb_file_path)
            merged_pdb = os.path.splitext(file_output_path)[0] + "_merged.pdb"
            pdb_structure = PDBParser(QUIET=True).get_structure("structure", pdb_file_path)
            contacts_ok = has_tcr_peptide_contact(pdb_structure, cutoff=10.0)
            print_verbose(f"Contacts (A/B to peptide <= 10 A): {contacts_ok}", verbose)

        elif file.endswith("summary_confidences.json"):
            summary_json_file_path = os.path.join(seed_folder_path, file)
            print_verbose(f"Summary JSON file: {summary_json_file_path}", verbose)

        elif file.endswith("confidences.json"):
            confidence_json_file_path = os.path.join(seed_folder_path, file)
            print_verbose(f"Confidence JSON file: {confidence_json_file_path}", verbose)

    if not (confidence_json_file_path and merged_pdb):
        print("Error processing metrics")
        return None

    sorted_removed_indices = get_sync_indices_for_pae(
        json_input_path, pdb_file_path, pdb_structure=pdb_structure)

    pae_output_path = os.path.join(folder_path, f"pae_{model_number}.npy")
    pde_output_path = os.path.join(folder_path, f"pde_{model_number}.npy")

    remove_atoms_from_pae_matrix(confidence_json_file_path, sorted_removed_indices, pae_output_path)
    remove_atoms_from_pde_matrix(confidence_json_file_path, sorted_removed_indices, pde_output_path)

    global_plddt = calculate_global_plddt(file_output_path, structure=cif_structure)
    (cdr1a_plddt, cdr2a_plddt, cdr3a_plddt,
     cdr1b_plddt, cdr2b_plddt, cdr3b_plddt,
     mean_peptide_plddt, mean_all_cdrs_peptide) = compute_cdr_metrics(
        file_output_path, "D", "E", structure=cif_structure)
    iptm_mean, chain_ptm_mean, iptm_tcrpmhc = calculate_iptms(summary_json_file_path)
    _, pdockq = calculate_pdockq(merged_pdb, pdockq_script)
    _, avgipde_A, avgipde_B, avgipae_A, avgipae_B, pdockq2_A, pdockq2_B = calculate_pdockq2(
        merged_pdb, pde_output_path, pae_output_path, pdockq2_script)
    ipsae, ipsae_d0chn, ipsae_d0dom, iptm_d0chn, lis = calculate_for_seed(
        seed_folder_path, model_file_path, confidence_json_file_path, scripts_dir=ipsae_scripts_dir)

    print_verbose(f"\nMetrics calculated for model {model_number} ---------------------------------------- ", verbose)
    print_verbose(f"Global PLDDT: {global_plddt}", verbose)
    print_verbose(f"CDR1s PLDDT: CDR1a {cdr1a_plddt}, CDR1b {cdr1b_plddt}, CDR2a {cdr2a_plddt}, "
                  f"CDR2b {cdr2b_plddt}, CDR3a {cdr3a_plddt}, CDR3b {cdr3b_plddt}", verbose)
    print_verbose(f"Mean peptide PLDDT: {mean_peptide_plddt}", verbose)
    print_verbose(f"Mean PLDDT of all CDRs and peptide: {mean_all_cdrs_peptide}", verbose)
    print_verbose(f"Chain IPTM mean: {iptm_mean}", verbose)
    print_verbose(f"Chain PTM mean: {chain_ptm_mean}", verbose)
    print_verbose(f"Interface TCR-pMHC IPTM mean: {iptm_tcrpmhc}", verbose)
    print_verbose(f"pDockQ: {pdockq}", verbose)
    print_verbose(f"Average iPDE A: {avgipde_A}, Average iPDE B: {avgipde_B}", verbose)
    print_verbose(f"Average iPAE A: {avgipae_A}, Average iPAE B: {avgipae_B}", verbose)
    print_verbose(f"pDockQ2 A: {pdockq2_A}, pDockQ2 B: {pdockq2_B}", verbose)
    print_verbose(f"ipSAE metrics ipSAE: {ipsae}, ipSAE d0chn {ipsae_d0chn}, ipSAE d0dom {ipsae_d0dom}, "
                  f"IPTM d0chn {iptm_d0chn}, LIS {lis}", verbose)
    print_verbose("------------------------------------------------------------------------", verbose)

    ranking_score = ranking_df.loc[ranking_df['model_number'] == int(model_number), 'ranking_score'].values

    row_data = {
        'tcr_id': tcr_id, 'model_number': model_number, 'seed': seed_number,
        'plddt': global_plddt, 'cdr1a_plddt': cdr1a_plddt, 'cdr1b_plddt': cdr1b_plddt,
        'cdr2a_plddt': cdr2a_plddt, 'cdr2b_plddt': cdr2b_plddt,
        'cdr3a_plddt': cdr3a_plddt, 'cdr3b_plddt': cdr3b_plddt,
        'peptide_plddt': mean_peptide_plddt, 'cdrs_peptide_plddt': mean_all_cdrs_peptide,
        'iptm_mean': iptm_mean, 'ptm_mean': chain_ptm_mean, 'iptm_tcrpmhc': iptm_tcrpmhc,
        'pdockq': pdockq, 'avgipde_a': avgipde_A, 'avgipde_b': avgipde_B,
        'avgipae_a': avgipae_A, 'avgipae_b': avgipae_B, 'pdockq2_a': pdockq2_A, 'pdockq2_b': pdockq2_B,
        'has_contacts': contacts_ok, 'ipsae': ipsae, 'ipsae_d0chn': ipsae_d0chn,
        'ipsae_d0dom': ipsae_d0dom, 'iptm_d0chn': iptm_d0chn,
        'lis': lis, 'ranking_score': ranking_score[0] if len(ranking_score) > 0 else None}

    os.remove(merged_pdb)
    return row_data


def process_tcr_folders(base_path, threshold=70, fast=False, verbose=False, seed_workers=4,
                         pdockq_script=None, pdockq2_script=None, ipsae_scripts_dir=None):
    if base_path.endswith("/"):
        basename = base_path.rstrip("/").split("/")[-1]
    else:
        basename = base_path.split("/")[-1]

    tcr_id = basename
    folder_path = base_path
    rows = []

    output_csv = os.path.join(folder_path, f"metrics_{tcr_id}.csv")

    json_input_path = next((f for f in glob.glob(os.path.join(folder_path, "*_data.json"))), None)
    if json_input_path is None:
        print(f"No *_data.json file found in {folder_path}")

    ranking_files = glob.glob(os.path.join(folder_path, "*ranking_scores.csv"))
    ranking_df = None
    if not ranking_files:
        print(f"No ranking_scores.csv file found in {folder_path}")
    else:
        ranking_df = pd.read_csv(ranking_files[0]).rename(columns={"sample": "model_number"})

    cols = ['tcr_id', 'model_number', 'seed', 'plddt', 'cdr1a_plddt', 'cdr1b_plddt', 'cdr2a_plddt', 'cdr2b_plddt',
            'cdr3a_plddt', 'cdr3b_plddt', 'peptide_plddt', 'cdrs_peptide_plddt', 'iptm_mean', 'ptm_mean',
            'iptm_tcrpmhc', 'pdockq', 'avgipde_b', 'avgipde_a', 'avgipae_a',
            'avgipae_b', 'pdockq2_a', 'pdockq2_b', 'has_contacts', 'ipsae', 'ipsae_d0chn', 'ipsae_d0dom',
            'iptm_d0chn', 'lis', 'ranking_score']

    if fast:
        print("Fast mode enabled")
        if os.path.exists(output_csv):
            print(f"CSV file {output_csv} already exists. Metrics will be parsed from this file without recalculating.")
            with open(output_csv, mode='r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                if not all(col in reader.fieldnames for col in cols):
                    print(f"CSV file {output_csv} is missing some required columns. Fast mode cannot be used "
                          f"with an incomplete metrics file in folder {folder_path}. Running in normal mode instead.")
                    fast = False
                else:
                    for row in reader:
                        rows.append(row)
        else:
            print(f"CSV file {output_csv} does not exist. Fast mode cannot be used without an existing metrics "
                  f"file in folder {folder_path}. Running in normal mode instead.")
            fast = False

    if not fast:
        if os.path.exists(output_csv):
            os.remove(output_csv)

        if ranking_df is None:
            print(f"Cannot proceed without ranking_scores.csv in {folder_path}")
            return pd.DataFrame(rows)

        seed_folders = [f for f in os.listdir(folder_path) if f.startswith("seed")]

        results = []
        with ThreadPoolExecutor(max_workers=max(1, seed_workers)) as executor:
            futures = {
                executor.submit(
                    process_single_seed, seed_folder, folder_path, tcr_id, threshold,
                    ranking_df, json_input_path, pdockq_script, pdockq2_script,
                    ipsae_scripts_dir, verbose
                ): seed_folder
                for seed_folder in seed_folders
            }
            for future in as_completed(futures):
                seed_folder = futures[future]
                try:
                    row_data = future.result()
                    if row_data is not None:
                        results.append(row_data)
                except Exception as e:
                    print(f"Error processing seed {seed_folder} in {folder_path}: {e}")

        with open(output_csv, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cols)
            writer.writeheader()
            for row_data in results:
                writer.writerow(row_data)
        rows = results

    metrics_df = pd.DataFrame(rows)
    return metrics_df

def run_one_folder(folder_path, output, threshold, fast, verbose, seed_workers,
                    pdockq_script, pdockq2_script, ipsae_scripts_dir):
    metrics_df = process_tcr_folders(
        folder_path, threshold=threshold, fast=fast, verbose=verbose, seed_workers=seed_workers,
        pdockq_script=pdockq_script, pdockq2_script=pdockq2_script, ipsae_scripts_dir=ipsae_scripts_dir)
    if metrics_df is None or len(metrics_df) == 0:
        print_verbose(f"[SKIP] {folder_path} (no data)", verbose)
        return
    file_exists = os.path.isfile(output)
    metrics_df.to_csv(output, mode="a", index=False, header=not file_exists)

def main():
    parser = argparse.ArgumentParser(description="Process TCR subfolders in parallel")
    parser.add_argument("base_path", type=str, help="Path containing TCR subfolders")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--threshold", type=int, default=70, help="PLDDT threshold for filtering residues")
    parser.add_argument("--fast", action="store_true",help="If set, skips some computationally expensive steps and uses existing metrics if available")
    parser.add_argument("--verbose", action="store_true", help="If set, prints verbose output")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 10)),help="Number of parallel worker PROCESSES, one per TCR folder")
    parser.add_argument("--seed-workers", type=int, default=4,help="Number of parallel THREADS used to process seeds within a single TCR folder. ")
    parser.add_argument("--pdockq-script", type=str,default="/gpfs/projects/bsc72/aascunce/programs/TCoaRse-ESMC/nextflow/src/pdockq.py")
    parser.add_argument("--pdockq2-script", type=str,default="/gpfs/projects/bsc72/aascunce/programs/TCoaRse-ESMC/nextflow/src/pdockq2_pae.py")
    parser.add_argument("--ipsae-scripts-dir", type=str,default="/gpfs/projects/bsc72/aascunce/programs/TCoaRse-ESMC/nextflow/src")
    args = parser.parse_args()

    subfolders = [os.path.join(args.base_path, d) for d in os.listdir(args.base_path)
                  if d.lower().startswith("tcr") and os.path.isdir(os.path.join(args.base_path, d))]
    print(f"Found {len(subfolders)} TCR folders")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one_folder, folder, args.output, args.threshold, args.fast, args.verbose, 
                                   args.seed_workers, args.pdockq_script, args.pdockq2_script, args.ipsae_scripts_dir,): folder for folder in subfolders}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing TCR folders"):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {folder}: {e}")


if __name__ == "__main__":
    main()