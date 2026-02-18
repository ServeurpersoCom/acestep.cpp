#!/usr/bin/env python3
"""Compare C++ vs Python detokenizer, step by step.

Runs dit-vae with --dump, then Python detokenizer, and compares.
Also validates Python intermediates against manual math to isolate bugs.

Usage:
    python3 tests/debug-detok-cos-sim.py <request.json>

The request must contain audio_codes (run ace-qwen3 first).
"""
import sys, os, json, struct, subprocess, shutil
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
GGML_BIN    = os.path.join(ROOT, "build", "dit-vae")
DIT_GGUF    = os.path.join(ROOT, "models", "acestep-v15-sft-BF16.gguf")
QWEN_GGUF   = os.path.join(ROOT, "models", "Qwen3-Embedding-0.6B-BF16.gguf")
VAE_GGUF    = os.path.join(ROOT, "models", "vae-BF16.gguf")

FSQ_LEVELS = [8, 8, 8, 5, 5, 5]

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0

def stats(name, a, b):
    c = cos(a, b)
    a_f, b_f = a.flatten(), b.flatten()
    n = min(len(a_f), len(b_f))
    diff = np.abs(a_f[:n] - b_f[:n])
    tag = "OK" if c > 0.999 else "BAD" if c < 0.99 else "WARN"
    print(f"  {name:25s} cos={c:.6f} maxdiff={diff.max():.6f} meandiff={diff.mean():.6f} [{tag}]")
    return c

def load_dump(path):
    raw = np.fromfile(path, dtype=np.float32)
    ndim = int(struct.unpack('i', struct.pack('f', raw[0]))[0])
    shape = [int(struct.unpack('i', struct.pack('f', raw[1+i]))[0])
             for i in range(ndim)]
    data = raw[1 + ndim:]
    return data, shape

def fsq_decode_index(index):
    out = np.zeros(6, dtype=np.float32)
    stride = 1
    for d in range(6):
        L = FSQ_LEVELS[d]
        level_idx = (index // stride) % L
        half_L = (L - 1) / 2.0
        out[d] = level_idx / half_L - 1.0
        stride *= L
    return out

def run_ggml(request_path, dump_dir):
    if not os.path.isfile(GGML_BIN):
        print(f"[GGML] binary not found: {GGML_BIN}")
        return False
    if os.path.isdir(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(dump_dir)
    cmd = [
        GGML_BIN,
        "--dit", DIT_GGUF,
        "--text-encoder", QWEN_GGUF,
        "--vae", VAE_GGUF,
        "--request", request_path,
        "--dump", dump_dir,
        "--output", os.path.join(dump_dir, "output.wav"),
    ]
    print(f"[GGML] Running dit-vae...")
    r = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    detok_path = os.path.join(dump_dir, "detok_output.bin")
    if not os.path.isfile(detok_path):
        print(f"[GGML] FAILED: no detok_output.bin (exit {r.returncode})")
        if r.stderr:
            for line in r.stderr.strip().split('\n')[-10:]:
                print(f"  {line}")
        return False
    print(f"[GGML] Done")
    return True

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <request.json>")
        return 1

    request_path = sys.argv[1]
    req = json.load(open(request_path))
    if 'audio_codes' not in req or not req['audio_codes']:
        print("ERROR: request has no audio_codes (run ace-qwen3 first)")
        return 1

    codes = [int(x) for x in req['audio_codes'].split(',')]
    T_5Hz = len(codes)
    print(f"[Input] {T_5Hz} codes, first 5: {codes[:5]}")

    # Step 1: Run GGML
    dump_dir = os.path.join(SCRIPT_DIR, "detok-dump")
    if not run_ggml(request_path, dump_dir):
        return 1

    ggml_data, ggml_shape = load_dump(os.path.join(dump_dir, "detok_output.bin"))
    T_25Hz = ggml_shape[0]
    ggml_out = ggml_data.reshape(T_25Hz, 64)
    print(f"[GGML] detok_output: [{T_25Hz}, 64]")

    # Step 2: Run Python
    print("\n[Python] Loading model...")
    import torch
    sys.path.insert(0, os.path.join(ROOT, '..', 'ACE-Step-1.5'))
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    handler.initialize_service(
        project_root=os.path.join(ROOT, 'checkpoints'),
        config_path='acestep-v15-sft',
        device='cuda',
    )
    model = handler.model
    detok = model.detokenizer

    codes_tensor = torch.tensor([codes], dtype=torch.long, device='cuda').unsqueeze(-1)

    with torch.no_grad():
        # FSQ dequant + project_out
        lm_hints_5Hz = model.tokenizer.quantizer.get_output_from_indices(codes_tensor)
        py_after_proj = lm_hints_5Hz[0].float().cpu().detach().numpy()

        # embed_tokens
        py_embedded = detok.embed_tokens(lm_hints_5Hz)
        py_embed_np = py_embedded[0].float().cpu().detach().numpy()

        # special_tokens + broadcast
        B, T, D = py_embedded.shape
        x = py_embedded.unsqueeze(2).repeat(1, 1, 5, 1)
        special = detok.special_tokens.expand(B, T, -1, -1)
        py_after_special = (x + special)[0, 0].float().cpu().detach().numpy()

        # Full detokenize
        lm_hints_25Hz = model.detokenize(lm_hints_5Hz)
        py_out = lm_hints_25Hz[0].float().cpu().detach().numpy()

    print(f"[Python] detok output: {py_out.shape}")

    # Step 3: GGML vs Python final comparison
    print(f"\n[Compare] GGML vs Python ({T_25Hz} frames)")
    n = min(len(ggml_out), len(py_out))
    stats("detok_output (full)", ggml_out[:n], py_out[:n])

    for t in range(min(5, T_5Hz)):
        g = ggml_out[t*5:(t+1)*5]
        p = py_out[t*5:(t+1)*5]
        stats(f"  token {t} (5 frames)", g, p)

    print(f"\n  Frame 0 (ch 0-7):")
    print(f"    GGML:   {ggml_out[0, :8]}")
    print(f"    Python: {py_out[0, :8]}")

    # Step 4: Validate Python math (isolate which stage could break C++)
    print(f"\n[Math validation] Python intermediates vs manual compute")

    # FSQ decode
    fsq_manual = np.array([fsq_decode_index(c) for c in codes])
    fsq_layer = model.tokenizer.quantizer.layers[0]
    idx_tensor = torch.tensor([[[codes[0]]]], dtype=torch.long, device='cuda')
    raw_fsq = fsq_layer.indices_to_codes(idx_tensor)
    raw_fsq_np = raw_fsq[0, 0, 0].float().cpu().detach().numpy()
    stats("FSQ decode tok0", fsq_manual[0], raw_fsq_np)

    # project_out
    proj_w = model.tokenizer.quantizer.project_out.weight.float().cpu().detach().numpy()
    proj_b = model.tokenizer.quantizer.project_out.bias.float().cpu().detach().numpy()
    manual_proj = fsq_manual[0] @ proj_w.T + proj_b
    stats("project_out tok0", manual_proj, py_after_proj[0])

    # embed_tokens
    embed_w = detok.embed_tokens.weight.float().cpu().detach().numpy()
    embed_b = detok.embed_tokens.bias.float().cpu().detach().numpy()
    manual_embed = py_after_proj[0] @ embed_w.T + embed_b
    stats("embed_tokens tok0", manual_embed, py_embed_np[0])

    # special_tokens
    special_np = detok.special_tokens[0].float().cpu().detach().numpy()
    manual_after_special = np.tile(manual_embed, (5, 1)) + special_np
    stats("special_tokens tok0", manual_after_special, py_after_special)

    print(f"\n[Summary]")
    c_final = cos(ggml_out[:n], py_out[:n])
    if c_final > 0.999:
        print(f"  PASS: cos={c_final:.6f}")
    elif c_final > 0.99:
        print(f"  WARN: cos={c_final:.6f} (precision issue, check bf16 vs f32)")
    else:
        print(f"  FAIL: cos={c_final:.6f}")
        print(f"  If math validation OK above, bug is in C++ 2L encoder (attn/MLP).")
        print(f"  If math validation BAD, check weight loading / FSQ / projections.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
