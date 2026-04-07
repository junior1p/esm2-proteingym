"""Minimal test: 5 aa sequence, 5 mutants only."""
import time
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.utils import generate_all_single_mutants, validate_sequence
from src.esm2_scorer import load_esm2_model, score_masked_marginal_batch

# Tiny test: 5 aa
seq = "ACDEF"
validate_sequence(seq)
mutants = generate_all_single_mutants(seq)
print(f"Seq: {seq} ({len(seq)} aa), {len(mutants)} mutants", flush=True)

model, tokenizer, device = load_esm2_model("facebook/esm2_t12_35M_UR50D")
print(f"Device: {device}", flush=True)

start = time.time()
results = score_masked_marginal_batch(seq, mutants, model, tokenizer, device, batch_size=32, progress_every=50)
elapsed = time.time() - start

print(f"\n{len(mutants)} mutants in {elapsed:.1f}s ({len(mutants)/elapsed:.1f}/sec)", flush=True)
for r in sorted(results, key=lambda x: x['esm2_score'], reverse=True):
    print(f"  {r['mutant']}: {r['esm2_score']:.4f}", flush=True)
