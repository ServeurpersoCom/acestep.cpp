#!/usr/bin/env python3
"""
test_lm_logits.py: compare GGML ace-qwen3 logits vs real HuggingFace model.

Uses AutoModelForCausalLM (same as ACE-Step LLMHandler internally).
NOT a from-scratch reimplementation -- this IS the ground truth.

Usage:
  1) Run GGML to dump logits/tokens:
     ./build/ace-qwen3 --model checkpoints/acestep-5Hz-lm-0.6B \
       --dump-logits /tmp/logits.bin --dump-tokens /tmp/tokens.csv \
       --caption "A dreamy ambient" --lyrics "[Instrumental]" \
       --output-dir /tmp/ace-out

  2) Compare against Python reference:
     python3 scripts/test_lm_logits.py checkpoints/acestep-5Hz-lm-0.6B \
       /tmp/logits.bin /tmp/tokens.csv
"""
import os, sys
import numpy as np


def python_logits(model_dir, prompt_tokens):
    """Forward pass with the real HF model (ground truth)."""
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[HF] Loading {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.cuda().eval()

    ids = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = model(ids)

    logits = out.logits[0, -1].float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return logits


def main():
    if len(sys.argv) < 4:
        print("Usage: test_lm_logits.py <model_dir> <ggml_logits.bin> <tokens.csv>")
        print("  1) ace-qwen3 --dump-logits logits.bin --dump-tokens tokens.csv ...")
        print("  2) python3 test_lm_logits.py checkpoints/acestep-5Hz-lm-0.6B logits.bin tokens.csv")
        return

    model_dir = sys.argv[1]
    ggml_logits_path = sys.argv[2]
    tokens_path = sys.argv[3]

    # Load GGML-dumped tokens
    with open(tokens_path, 'r') as f:
        prompt_tokens = [int(x) for x in f.read().strip().split(',')]

    print(f"Prompt: {len(prompt_tokens)} tokens")
    print(f"First 10: {prompt_tokens[:10]}")
    print(f"Last 10:  {prompt_tokens[-10:]}")

    # HF reference (ground truth)
    pt_logits = python_logits(model_dir, prompt_tokens)

    print(f"\nHF logits:  min={pt_logits.min():.4f} max={pt_logits.max():.4f}")
    print(f"HF argmax:  {pt_logits.argmax()} (val={pt_logits.max():.4f})")
    print(f"HF top5:    {np.argsort(pt_logits)[-5:][::-1]}")

    # GGML logits
    with open(ggml_logits_path, 'rb') as f:
        ggml_logits = np.frombuffer(f.read(), dtype=np.float32)

    print(f"GGML logits: min={ggml_logits.min():.4f} max={ggml_logits.max():.4f}")
    print(f"GGML argmax: {ggml_logits.argmax()} (val={ggml_logits.max():.4f})")
    print(f"GGML top5:   {np.argsort(ggml_logits)[-5:][::-1]}")

    # Cosine similarity
    dot = np.dot(pt_logits, ggml_logits)
    cos = dot / (np.linalg.norm(pt_logits) * np.linalg.norm(ggml_logits) + 1e-12)
    print(f"\nCosine similarity HF<>GGML: {cos:.6f}")

    # Top-k overlap
    pt_top10 = set(np.argsort(pt_logits)[-10:])
    gg_top10 = set(np.argsort(ggml_logits)[-10:])
    print(f"Top-10 overlap: {len(pt_top10 & gg_top10)}/10")

    # Max absolute difference
    maxdiff = np.max(np.abs(pt_logits - ggml_logits))
    print(f"Max abs diff: {maxdiff:.6f}")

    # Verdict
    if cos > 0.999:
        print("\nVERDICT: EXCELLENT match")
    elif cos > 0.99:
        print("\nVERDICT: Good match (minor precision diffs)")
    elif cos > 0.95:
        print("\nVERDICT: ACCEPTABLE (check bf16/f32 precision)")
    else:
        print("\nVERDICT: MISMATCH -- investigate!")


if __name__ == "__main__":
    main()
