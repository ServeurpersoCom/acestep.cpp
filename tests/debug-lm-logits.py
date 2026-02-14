#!/usr/bin/env python3
"""Compare first-token logits: GGML vs PyTorch for ace-qwen3 LM"""
import sys, struct, json, os
import numpy as np

# Load safetensors + run one forward pass in PyTorch
def test_pytorch_logits(model_dir, prompt_tokens):
    import torch
    from safetensors.torch import load_file

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    # Load weights
    st_path = os.path.join(model_dir, "model.safetensors")
    weights = load_file(st_path)

    H = cfg["hidden_size"]
    V = cfg["vocab_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    inter = cfg["intermediate_size"]
    rope_theta = cfg.get("rope_theta", 1000000.0)
    eps = cfg.get("rms_norm_eps", 1e-6)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # match GGML f32 compute

    # Move weights to device
    for k in weights:
        weights[k] = weights[k].to(device=device, dtype=dtype)

    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    S = tokens.shape[1]

    # Embedding
    hidden = weights["model.embed_tokens.weight"][tokens[0]]  # [S, H]

    # Positions
    positions = torch.arange(S, device=device)

    # Precompute RoPE freqs
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = positions.float()
    freqs = torch.outer(t, freqs)  # [S, D/2]
    cos_f = torch.cos(freqs)
    sin_f = torch.sin(freqs)

    def rms_norm(x, w):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return (x / rms) * w

    def apply_rope(x, cos_f, sin_f):
        # x: [S, Nh, D] -> NEOX layout
        D = x.shape[-1]
        x1 = x[..., :D//2]
        x2 = x[..., D//2:]
        # Broadcast cos/sin: [S, 1, D/2]
        c = cos_f.unsqueeze(1)
        s = sin_f.unsqueeze(1)
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    # Causal mask
    mask = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1)

    for l in range(n_layers):
        prefix = f"model.layers.{l}"

        # Pre-attn norm
        normed = rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"])

        # QKV
        q = normed @ weights[f"{prefix}.self_attn.q_proj.weight"].T  # [S, Nh*D]
        k = normed @ weights[f"{prefix}.self_attn.k_proj.weight"].T  # [S, Nkv*D]
        v = normed @ weights[f"{prefix}.self_attn.v_proj.weight"].T  # [S, Nkv*D]

        q = q.view(S, n_heads, head_dim)
        k = k.view(S, n_kv_heads, head_dim)
        v = v.view(S, n_kv_heads, head_dim)

        # QK-norm
        q = rms_norm(q, weights[f"{prefix}.self_attn.q_norm.weight"])
        k = rms_norm(k, weights[f"{prefix}.self_attn.k_norm.weight"])

        # RoPE
        q = apply_rope(q, cos_f, sin_f)
        k = apply_rope(k, cos_f, sin_f)

        # GQA: expand KV heads
        rep = n_heads // n_kv_heads
        if rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, rep, -1).reshape(S, n_heads, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, rep, -1).reshape(S, n_heads, head_dim)

        # Attention: [S, Nh, D] -> [Nh, S, D]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        scale = 1.0 / (head_dim ** 0.5)
        attn_w = torch.matmul(q, k.transpose(-1, -2)) * scale + mask
        attn_w = torch.softmax(attn_w, dim=-1)
        attn_out = torch.matmul(attn_w, v)  # [Nh, S, D]
        attn_out = attn_out.transpose(0, 1).reshape(S, n_heads * head_dim)  # [S, Nh*D]

        # O proj
        attn_out = attn_out @ weights[f"{prefix}.self_attn.o_proj.weight"].T

        # Residual
        hidden = hidden + attn_out

        # Post-attn norm + MLP
        normed = rms_norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"])
        gate = normed @ weights[f"{prefix}.mlp.gate_proj.weight"].T
        up = normed @ weights[f"{prefix}.mlp.up_proj.weight"].T
        mlp_out = (torch.nn.functional.silu(gate) * up)
        mlp_out = mlp_out @ weights[f"{prefix}.mlp.down_proj.weight"].T
        hidden = hidden + mlp_out

    # Final norm
    hidden = rms_norm(hidden, weights["model.norm.weight"])

    # Logits (last token)
    logits = hidden[-1] @ weights["model.embed_tokens.weight"].T  # [V]
    return logits.cpu().numpy()

def main():
    if len(sys.argv) < 4:
        print("Usage: debug-lm-logits.py <model_dir> <ggml_logits.bin> <tokens.csv>")
        print("  1) ace-qwen3 --dump-logits logits.bin --dump-tokens tokens.csv ...")
        print("  2) python3 tests/debug-lm-logits.py checkpoints/acestep-5Hz-lm-0.6B logits.bin tokens.csv")
        return

    model_dir = sys.argv[1]
    ggml_logits_path = sys.argv[2]
    tokens_path = sys.argv[3]

    with open(tokens_path, 'r') as f:
        prompt_tokens = [int(x) for x in f.read().strip().split(',')]

    print(f"[Test] Prompt: {len(prompt_tokens)} tokens, first 10: {prompt_tokens[:10]}")

    # PyTorch reference
    pt_logits = test_pytorch_logits(model_dir, prompt_tokens)

    print(f"[Python] logits: min={pt_logits.min():.4f} max={pt_logits.max():.4f}")
    print(f"[Python] argmax: {pt_logits.argmax()} (val={pt_logits.max():.4f})")
    print(f"[Python] top5: {np.argsort(pt_logits)[-5:][::-1]}")

    # GGML logits
    if ggml_logits_path and os.path.exists(ggml_logits_path):
        with open(ggml_logits_path, 'rb') as f:
            ggml_logits = np.frombuffer(f.read(), dtype=np.float32)

        print(f"[GGML] logits: min={ggml_logits.min():.4f} max={ggml_logits.max():.4f}")
        print(f"[GGML] argmax: {ggml_logits.argmax()} (val={ggml_logits.max():.4f})")
        print(f"[GGML] top5: {np.argsort(ggml_logits)[-5:][::-1]}")

        # Cosine similarity
        dot = np.dot(pt_logits, ggml_logits)
        norm_pt = np.linalg.norm(pt_logits)
        norm_gg = np.linalg.norm(ggml_logits)
        cos = dot / (norm_pt * norm_gg + 1e-12)
        print(f"[Test] Cosine similarity Python<>GGML: {cos:.6f}")

        # Top-k agreement
        pt_top10 = set(np.argsort(pt_logits)[-10:])
        gg_top10 = set(np.argsort(ggml_logits)[-10:])
        print(f"[Test] Top-10 overlap: {len(pt_top10 & gg_top10)}/10")

if __name__ == "__main__":
    main()
