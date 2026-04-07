"""ESM-2 masked marginal scoring for zero-shot mutation fitness prediction.

Core algorithm: masked marginal log-likelihood ratio (LLR)
  score(X_i → Y_i) = log p(Y_i | x_{-i}) − log p(X_i | x_{-i})

This is the best-performing zero-shot scoring strategy for ESM models
(Meier et al., 2021; outperforms wild-type marginal and PPPL).
"""

import torch
import numpy as np
from transformers import EsmTokenizer, EsmForMaskedLM
from .utils import generate_all_single_mutants

MODEL_MAP = {
    "35M":  "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B":   "facebook/esm2_t36_3B_UR50D",
}


def load_esm2_model(model_name: str = "facebook/esm2_t33_650M_UR50D"):
    """Load ESM-2 model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def score_masked_marginal(
    sequence: str,
    position: int,
    mut_aa: str,
    model,
    tokenizer,
    device: str
) -> float:
    """Compute masked marginal LLR for a single mutant at a given position.
    
    score = log p(mut_aa | context) - log p(wt_aa | context)
    Positive = likely beneficial; Negative = likely deleterious.
    """
    wt_aa = sequence[position]
    
    # Create masked sequence: replace position with <mask>
    masked_seq = sequence[:position] + "<mask>" + sequence[position+1:]
    
    # Tokenize
    inputs = tokenizer(masked_seq, return_tensors="pt").to(device)
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, seq_len, vocab_size)
    log_probs = torch.nn.functional.log_softmax(logits[0, mask_positions[0]], dim=-1)
    
    wt_token_id = tokenizer.convert_tokens_to_ids(wt_aa)
    mut_token_id = tokenizer.convert_tokens_to_ids(mut_aa)
    
    llr = (log_probs[0, mut_token_id] - log_probs[0, wt_token_id]).item()
    return llr


def score_all_mutants(
    sequence: str,
    mutants: list[dict],
    model,
    tokenizer,
    device: str,
    batch_report_every: int = 100
) -> list[dict]:
    """Score all mutants using ESM-2 masked marginal scoring."""
    results = []
    for i, m in enumerate(mutants):
        score = score_masked_marginal(
            sequence=sequence,
            position=m["position"] - 1,  # convert to 0-indexed
            mut_aa=m["mut_aa"],
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        results.append({**m, "esm2_score": score})
        if (i + 1) % batch_report_every == 0:
            print(f"  Scored {i+1}/{len(mutants)} mutants...")
    return results


def run_scorer(sequence: str, model_size: str = "650M") -> tuple[list[dict], str, str]:
    """Main entry point for ESM-2 scoring.
    
    Returns:
        predictions: list of dicts with mutant info and esm2_score
        model_name: full model identifier
        device: cpu/cuda
    """
    model_name = MODEL_MAP.get(model_size, MODEL_MAP["650M"])
    model, tokenizer, device = load_esm2_model(model_name)
    
    mutants = generate_all_single_mutants(sequence)
    print(f"Generated {len(mutants)} single-point mutants for sequence of length {len(sequence)}")
    
    predictions = score_all_mutants(sequence, mutants, model, tokenizer, device)
    print(f"Scoring complete. Total mutants scored: {len(predictions)}")
    
    return predictions, model_name, device
