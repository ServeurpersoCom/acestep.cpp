#!/usr/bin/env python3
"""GGML vs Python cosine similarity comparison for dit-vae pipeline.

Runs both backends with identical inputs (caption, lyrics, noise, seed),
dumps intermediate tensors at each stage, and compares via cosine similarity.

Usage:
    python3 tests/debug-cos-sim.py                    # both backends
    python3 tests/debug-cos-sim.py --skip-python      # GGML only (dump)
    python3 tests/debug-cos-sim.py --skip-ggml        # Python only (dump)
    python3 tests/debug-cos-sim.py --caption "..." --bpm 90 --duration 60
"""
import argparse, os, struct, subprocess, shutil, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
ACE_STEP   = os.path.join(ROOT, "..", "ACE-Step-1.5")

GGML_BIN   = os.path.join(ROOT, "build", "dit-vae")
CKPT_DIR   = os.path.join(ROOT, "checkpoints", "acestep-v15-turbo")
VAE_DIR    = os.path.join(ROOT, "checkpoints", "vae")
QWEN_DIR   = os.path.join(ROOT, "checkpoints", "Qwen3-Embedding-0.6B")

DUMP_GGML   = os.path.join(SCRIPT_DIR, "ggml")
DUMP_PYTHON = os.path.join(SCRIPT_DIR, "python")

CAPTION  = "A dreamy ambient electronic track with soft synths and gentle pads"
LYRICS   = "[Instrumental]"
BPM      = 120
DURATION = 30
SEED     = 42
T_MAX    = 8500  # 340s * 25Hz, covers any duration up to 340s

def generate_philox_noise(path, seed=42, T=T_MAX, C=64):
    """Generate Philox CUDA RNG noise and save as raw bf16.

    Matches PyTorch handler.py: torch.randn([1, C, T], generator=Philox(seed), device='cuda').
    File layout: [C, T] contiguous bf16 (same as PyTorch memory order).
    """
    try:
        import torch
    except ImportError:
        print(f"[Noise] ERROR: torch required to generate {path}")
        return False
    if not torch.cuda.is_available():
        print(f"[Noise] ERROR: CUDA required for Philox RNG (generates different values on CPU)")
        return False
    gen = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn([1, C, T], generator=gen, device="cuda", dtype=torch.bfloat16)
    raw = noise[0].cpu().contiguous().view(torch.uint8).numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)
    print(f"[Noise] Generated {path}: [{C}, {T}] bf16 ({len(raw)} bytes, seed={seed})")
    return True

# binary dump I/O (matches C++ format: [ndim i32] [shape i32...] [data f32])

def save_dump(path, data):
    if hasattr(data, "detach"):
        data = data.detach().float().cpu().numpy()
    if hasattr(data, "numpy"):
        data = data.numpy()
    data = np.ascontiguousarray(data, dtype=np.float32)
    shape = list(data.shape)
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(shape)))
        for s in shape:
            f.write(struct.pack("i", s))
        f.write(data.tobytes())

def load_dump(path):
    raw = np.fromfile(path, dtype=np.float32)
    ndim = int(struct.unpack("i", struct.pack("f", raw[0]))[0])
    return raw[1 + ndim:]

def cos(a, b):
    a, b = a.flatten(), b.flatten()
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0

def log_spectral_cos(a, b, n_bands=80):
    n = min(len(a), len(b)) // 2
    if n < 1024:
        return 0.0
    fa, fb = np.abs(np.fft.rfft(a[:n])), np.abs(np.fft.rfft(b[:n]))
    bs = len(fa) // n_bands
    if bs < 1:
        return 0.0
    la = np.log1p(np.array([np.sum(fa[i*bs:(i+1)*bs]**2) for i in range(n_bands)]))
    lb = np.log1p(np.array([np.sum(fb[i*bs:(i+1)*bs]**2) for i in range(n_bands)]))
    d = np.linalg.norm(la) * np.linalg.norm(lb)
    return float(np.dot(la, lb) / d) if d > 1e-10 else 0.0

# GGML runner

def write_metadata(dump_dir, caption, lyrics, bpm, duration):
    for name, val in [("caption", caption), ("lyrics", lyrics),
                      ("bpm", str(bpm)), ("duration", str(duration))]:
        with open(os.path.join(dump_dir, name), "w") as f:
            f.write(val)

def run_ggml(dump_dir, caption, lyrics, bpm, duration, seed, noise_file=None):
    if not os.path.isfile(GGML_BIN):
        print(f"[GGML] binary not found: {GGML_BIN}")
        return False
    os.makedirs(dump_dir, exist_ok=True)
    write_metadata(dump_dir, caption, lyrics, bpm, duration)
    cmd = [
        GGML_BIN,
        "--dit", CKPT_DIR, "--vae", VAE_DIR, "--text-encoder", QWEN_DIR,
        "--input-dir", dump_dir,
        "--seed", str(seed),
        "--dump", dump_dir,
        "--output", os.path.join(dump_dir, "output.wav"),
    ]
    if noise_file:
        cmd += ["--noise-file", noise_file]
    print("[GGML] running...")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    if r.returncode != 0:
        print(f"[GGML] FAILED (exit {r.returncode})")
        if r.stdout:
            print(r.stdout[-500:])
        return False
    n = len([f for f in os.listdir(dump_dir) if f.endswith(".bin")])
    print(f"[GGML] done, {n} dump files")
    return True

# Python runner

def run_python(dump_dir, caption, lyrics, bpm, duration, seed):
    os.makedirs(dump_dir, exist_ok=True)

    sys.path.insert(0, ACE_STEP)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import torch
    from acestep.handler import AceStepHandler

    print("[Python] initializing handler...")
    handler = AceStepHandler()
    msg, ok = handler.initialize_service(
        project_root=ACE_STEP,
        config_path="acestep-v15-turbo",
        device="cuda",
    )
    if not ok:
        print(f"[Python] init failed: {msg}")
        return False
    print(f"[Python] init ok")

    _dumps = {}

    # Hook 1: text_hidden (text encoder output)
    orig_text_emb = handler.infer_text_embeddings
    def hooked_text_emb(token_ids):
        out = orig_text_emb(token_ids)
        _dumps["text_hidden"] = out[0].clone()
        return out
    handler.infer_text_embeddings = hooked_text_emb

    # Hook 2: lyric_embed (vocab lookup)
    orig_lyric = handler.infer_lyric_embeddings
    def hooked_lyric(token_ids):
        out = orig_lyric(token_ids)
        _dumps["lyric_embed"] = out[0].clone()
        return out
    handler.infer_lyric_embeddings = hooked_lyric

    # Hook 3: prepare_condition -> enc_hidden, context
    model = handler.model
    orig_prepare = model.prepare_condition
    def hooked_prepare(*args, **kwargs):
        enc_hs, enc_am, ctx = orig_prepare(*args, **kwargs)
        _dumps["enc_hidden"] = enc_hs[0].clone()
        _dumps["context"] = ctx[0].clone()
        return enc_hs, enc_am, ctx
    model.prepare_condition = hooked_prepare

    # Hook 4: noise (from prepare_noise)
    orig_noise = model.prepare_noise
    def hooked_noise(*args, **kwargs):
        n = orig_noise(*args, **kwargs)
        _dumps["noise"] = n[0].clone()
        return n
    model.prepare_noise = hooked_noise

    # Hook 5: decoder.forward -> dit_step*_vt
    decoder = model.decoder
    _step = [0]
    orig_fwd = decoder.forward
    def hooked_fwd(*args, **kwargs):
        out = orig_fwd(*args, **kwargs)
        vt = out[0]
        _dumps[f"dit_step{_step[0]}_vt"] = vt[0].clone()
        _step[0] += 1
        return out
    decoder.forward = hooked_fwd

    print("[Python] generating...")
    result = handler.generate_music(
        captions=caption,
        lyrics=lyrics,
        bpm=bpm,
        audio_duration=float(duration),
        seed=str(seed),
        use_random_seed=False,
        batch_size=1,
        inference_steps=8,
        shift=3.0,
        infer_method="ode",
        vocal_language="en",
    )

    if not result.get("success"):
        print(f"[Python] generation failed: {result.get('error', 'unknown')}")
        return False

    extra = result.get("extra_outputs", {})
    if extra.get("pred_latents") is not None:
        _dumps["dit_x0"] = extra["pred_latents"][0]

    audios = result.get("audios", [])
    if audios and "tensor" in audios[0]:
        _dumps["vae_audio"] = audios[0]["tensor"].squeeze(0)

    for name, tensor in sorted(_dumps.items()):
        save_dump(os.path.join(dump_dir, f"{name}.bin"), tensor)

    print(f"[Python] done, {len(_dumps)} dump files: {sorted(_dumps.keys())}")
    return True

# comparison

STAGES = [
    "text_hidden", "lyric_embed", "enc_hidden", "context", "noise",
    "dit_step0_vt", "dit_step1_vt", "dit_step2_vt", "dit_step3_vt",
    "dit_step4_vt", "dit_step5_vt", "dit_step6_vt", "dit_step7_vt",
    "dit_x0", "vae_audio",
]

def compare(dirs):
    labels = list(dirs.keys())
    pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i+1:]]

    print(f"\n{'stage':30s}", end="")
    for a, b in pairs:
        tag = f"{a}<>{b}"
        print(f" {tag:>14s}", end="")
    print()
    print("-" * (30 + 15 * len(pairs)))

    for stage in STAGES:
        data = {}
        for label, d in dirs.items():
            f = os.path.join(d, stage + ".bin")
            if os.path.isfile(f):
                data[label] = load_dump(f)

        print(f"{stage:30s}", end="")
        for a, b in pairs:
            if a in data and b in data:
                c = cos(data[a], data[b])
                print(f" {c:>14.6f}", end="")
            else:
                print(f" {'N/A':>14s}", end="")
        print()

    vae_data = {}
    for label, d in dirs.items():
        f = os.path.join(d, "vae_audio.bin")
        if os.path.isfile(f):
            vae_data[label] = load_dump(f)
    if len(vae_data) >= 2:
        print(f"\n{'vae_audio (log spectral)':30s}", end="")
        for a, b in pairs:
            if a in vae_data and b in vae_data:
                c = log_spectral_cos(vae_data[a], vae_data[b])
                print(f" {c:>14.6f}", end="")
            else:
                print(f" {'N/A':>14s}", end="")
        print()

# main

def main():
    ap = argparse.ArgumentParser(description="GGML vs Python cosine similarity comparison")
    ap.add_argument("--skip-ggml",   action="store_true", help="skip GGML backend")
    ap.add_argument("--skip-python", action="store_true", help="skip Python backend")
    ap.add_argument("--caption",     default=CAPTION)
    ap.add_argument("--lyrics",      default=LYRICS)
    ap.add_argument("--bpm",         type=int, default=BPM)
    ap.add_argument("--duration",    type=int, default=DURATION)
    ap.add_argument("--seed",        type=int, default=SEED)
    ap.add_argument("--noise-file",  default=None,
                    help="bf16 noise file for GGML (default: auto for seed 42)")
    args = ap.parse_args()

    noise_file = args.noise_file
    if noise_file is None and args.seed == 42:
        auto = os.path.join(SCRIPT_DIR, "rng_philox_seed42.bf16")
        if not os.path.isfile(auto):
            print(f"[Noise] {auto} not found, generating...")
            if not generate_philox_noise(auto, seed=42):
                print("[Noise] WARNING: cannot generate noise file, GGML will use mt19937 (results will differ)")
        if os.path.isfile(auto):
            noise_file = auto

    dirs = {}

    if not args.skip_ggml:
        if os.path.isdir(DUMP_GGML):
            shutil.rmtree(DUMP_GGML)
        print("\n[Test] GGML backend")
        if run_ggml(DUMP_GGML, args.caption, args.lyrics,
                    args.bpm, args.duration, args.seed,
                    noise_file=noise_file):
            dirs["GGML"] = DUMP_GGML

    if not args.skip_python:
        if os.path.isdir(DUMP_PYTHON):
            shutil.rmtree(DUMP_PYTHON)
        print("\n[Test] Python backend")
        if run_python(DUMP_PYTHON, args.caption, args.lyrics, args.bpm,
                      args.duration, args.seed):
            dirs["Python"] = DUMP_PYTHON

    if len(dirs) < 2:
        print(f"[Test] Need both backends, got {len(dirs)}: {list(dirs.keys())}")
        return 1

    print("\n[Test] Comparison GGML vs Python")
    compare(dirs)
    return 0

if __name__ == "__main__":
    sys.exit(main())
