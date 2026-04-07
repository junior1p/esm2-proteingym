"""Utility functions for ESM-2 mutation scoring."""

import requests
from typing import Optional

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
VALID_AA = set(AMINO_ACIDS)


def fetch_uniprot_sequence(uniprot_id: str) -> str:
    """Fetch canonical protein sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = response.text.strip().split('\n')
    return ''.join(lines[1:])  # skip FASTA header


def validate_sequence(sequence: str) -> None:
    """Validate sequence contains only standard 20 amino acids."""
    invalid = set(sequence.upper()) - VALID_AA
    if invalid:
        raise ValueError(
            f"Invalid amino acids in sequence: {invalid}. "
            f"Please use only: {''.join(sorted(VALID_AA))}"
        )


def generate_all_single_mutants(sequence: str) -> list[dict]:
    """Generate all possible single amino acid substitutions.
    
    Returns list of dicts with keys:
        mutant (str), position (int), wt_aa (str), mut_aa (str), mutated_sequence (str)
    """
    mutants = []
    for i, wt_aa in enumerate(sequence):
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue  # skip synonymous
            mutant_label = f"{wt_aa}{i+1}{mut_aa}"  # e.g. "A42G"
            mutated_seq = sequence[:i] + mut_aa + sequence[i+1:]
            mutants.append({
                "mutant": mutant_label,
                "position": i + 1,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "mutated_sequence": mutated_seq
            })
    return mutants
