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
    print(f"Loading {model_name}...", flush=True)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}", flush=True)
    return model, tokenizer, device


def score_masked_marginal_batch(
    sequence: str,
    mutants: list[dict],
    model,
    tokenizer,
    device: str,
    batch_size: int = 1,
    progress_every: int = 100
) -> list[dict]:
    """Score all mutants using per-mutant masked marginal scoring with no_grad forward.
    
    Each mutant gets its own forward pass (batch_size=1 to avoid padding overhead).
    With ESM-2 35M on CPU: ~1-3 sec/mutant → GFP 4522 mutants ≈ 1.5-4.5 hours.
    
    Use batch_size > 1 only if you have a GPU (avoids CPU padding overhead).
    """
    n = len(sequence)
    
    # Compute WT log-probs: one forward per position
    print("Computing wild-type log-probs...", flush=True)
    wt_log_probs = []
    for i in range(n):
        masked_seq = sequence[:i] + "<mask>" + sequence[i+1:]
        inputs = tokenizer(masked_seq, return_tensors="pt").to(device)
        mask_token_id = tokenizer.mask_token_id
        mask_pos = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[0]
        
        with torch.no_grad():
            logits = model(**inputs).logits
        # logits shape: (1, seq_len, vocab_size); mask_pos[0] is the position
        log_probs = torch.nn.functional.log_softmax(logits[0, mask_pos[0]], dim=-1)
        wt_token_id = tokenizer.convert_tokens_to_ids(sequence[i])
        wt_log_probs.append(log_probs[wt_token_id].item())
        
        if (i + 1) % 50 == 0:
            print(f"  WT pos {i+1}/{n}...", flush=True)
    
    print(f"WT log-probs computed for {n} positions", flush=True)
    
    # Score all mutants
    results = []
    total = len(mutants)
    scored = 0
    
    for m in mutants:
        pos = m["position"] - 1
        masked_seq = sequence[:pos] + "<mask>" + sequence[pos+1:]
        inputs = tokenizer(masked_seq, return_tensors="pt").to(device)
        mask_token_id = tokenizer.mask_token_id
        mask_pos = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[0]
        
        with torch.no_grad():
            logits = model(**inputs).logits
        log_probs = torch.nn.functional.log_softmax(logits[0, mask_pos[0]], dim=-1)
        mut_token_id = tokenizer.convert_tokens_to_ids(m["mut_aa"])
        llr = (log_probs[mut_token_id] - wt_log_probs[pos]).item()
        results.append({**m, "esm2_score": llr})
        
        scored += 1
        if scored % progress_every == 0:
            pct = scored / total * 100
            print(f"  Scored {scored}/{total} ({pct:.0f}%)...", flush=True)
    
    return results


def score_all_mutants(
    sequence: str,
    mutants: list[dict],
    model,
    tokenizer,
    device: str,
    batch_report_every: int = 100
) -> list[dict]:
    """Score all mutants using ESM-2 masked marginal scoring."""
    return score_masked_marginal_batch(
        sequence, mutants, model, tokenizer, device,
        batch_size=1, progress_every=batch_report_every
    )


def run_scorer(sequence: str, model_size: str = "650M") -> tuple[list[dict], str, str]:
    """Main entry point for ESM-2 scoring."""
    model_name = MODEL_MAP.get(model_size, MODEL_MAP["650M"])
    model, tokenizer, device = load_esm2_model(model_name)
    
    mutants = generate_all_single_mutants(sequence)
    print(f"Generated {len(mutants)} single-point mutants for sequence of length {len(sequence)}", flush=True)
    
    predictions = score_all_mutants(sequence, mutants, model, tokenizer, device)
    print(f"Scoring complete. Total mutants scored: {len(predictions)}", flush=True)
    
    return predictions, model_name, device
