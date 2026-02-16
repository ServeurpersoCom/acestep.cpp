#!/usr/bin/env python3
"""GGML vs Python cosine similarity comparison for dit-vae pipeline.

Supports both turbo (no CFG) and SFT (CFG + APG) models.

Usage:
    python3 tests/debug-cos-sim.py                        # both modes
    python3 tests/debug-cos-sim.py --mode turbo            # turbo only
    python3 tests/debug-cos-sim.py --mode sft              # SFT only
    python3 tests/debug-cos-sim.py --skip-python           # GGML only
    python3 tests/debug-cos-sim.py --skip-ggml             # Python only
"""
import os, sys, subprocess, struct, shutil, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
GGML_BIN   = os.path.join(ROOT, "build", "dit-vae")
VAE_DIR    = os.path.join(ROOT, "checkpoints", "vae")
QWEN_DIR   = os.path.join(ROOT, "checkpoints", "Qwen3-Embedding-0.6B")

# Per-mode config
MODE_CONFIG = {
    "turbo": {
        "ckpt_dir": os.path.join(ROOT, "checkpoints", "acestep-v15-turbo"),
        "config_path": "acestep-v15-turbo",
        "steps": 8, "shift": 3.0, "guidance": 0.0,
    },
    "sft": {
        "ckpt_dir": os.path.join(ROOT, "checkpoints", "acestep-v15-sft"),
        "config_path": "acestep-v15-sft",
        "steps": 50, "shift": 1.0, "guidance": 7.0,
    },
}

# Defaults
CAPTION  = "Upbeat pop rock anthem with driving electric guitars, punchy drums, catchy vocal hooks, and a singalong chorus"
LYRICS   = "[verse]\nWe ride the lightning through the night\nChasing echoes burning bright\n[chorus]\nWe are the fire we are the flame\nNothing will ever be the same"
BPM      = 120
DURATION = 120
SEED     = 42

def generate_philox_noise(path, seed=42, T=750, C=64):
    """Generate Philox CUDA RNG noise and save as raw bf16.

    CUDA Philox produces identical raw bytes regardless of logical shape:
    torch.randn([1, C, T]) == torch.randn([1, T, C]) in raw memory.
    File: 48000 bf16 values.  C++ reads linearly as [T, C] time-major,
    matching Python handler convention.
    """
    try:
        import torch
    except ImportError:
        print("[Noise] ERROR: PyTorch required")
        return False
    if not torch.cuda.is_available():
        print("[Noise] ERROR: CUDA required for Philox RNG (generates different values on CPU)")
        return False
    gen = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn([1, C, T], generator=gen, device="cuda", dtype=torch.bfloat16)
    raw = noise[0].cpu().contiguous().view(torch.uint8).numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)
    print(f"[Noise] Generated {path}: [{T}, {C}] bf16 ({len(raw)} bytes, seed={seed})")
    return True

# binary dump I/O (matches C++ format: [ndim i32] [shape i32...] [data f32])

def save_dump(path, data):
    import torch
    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu().numpy()
    data = np.ascontiguousarray(data.astype(np.float32))
    shape = data.shape
    header = struct.pack("i", len(shape))
    for s in shape:
        header += struct.pack("i", s)
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())

def load_dump(path):
    raw = np.fromfile(path, dtype=np.float32)
    ndim = int(struct.unpack("i", struct.pack("f", raw[0]))[0])
    shape = [int(struct.unpack("i", struct.pack("f", raw[1+i]))[0]) for i in range(ndim)]
    data = raw[1 + ndim:]
    return data, shape

def _cos_flat(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0

def cos(a, b, shape_a=None, shape_b=None):
    """Cosine similarity with automatic layout detection.

    GGML dumps with header [ne[0], ne[1]] where ne[0] is contiguous (Fortran-order).
    Python/numpy uses C-order where last dim is contiguous.
    For 2D: ggml [ne0, ne1] data should be reshaped as [ne1, ne0] in C-order.
    We try all layout combinations and return the best match.
    """
    c_flat = _cos_flat(a.flatten(), b.flatten())
    if shape_a and shape_b and len(shape_a) == 2 and len(shape_b) == 2:
        total_a, total_b = shape_a[0] * shape_a[1], shape_b[0] * shape_b[1]
        n = min(len(a), len(b), total_a, total_b)
        if total_a == total_b and n == total_a:
            best = c_flat
            try:
                rb = b[:n].reshape(shape_b)
                # Try: a is ggml (reversed dims), b is numpy (C-order)
                ra_rev = a[:n].reshape(shape_a[1], shape_a[0])  # ggml: [ne1, ne0]
                best = max(best, _cos_flat(ra_rev.T.flatten(), rb.flatten()))
                # Try: b is ggml (reversed dims), a is numpy (C-order)
                ra = a[:n].reshape(shape_a)
                rb_rev = b[:n].reshape(shape_b[1], shape_b[0])
                best = max(best, _cos_flat(ra.flatten(), rb_rev.T.flatten()))
            except (ValueError, RuntimeError):
                pass
            return best
    return c_flat

def log_spectral_cos(a, b, n_bands=80):
    n = min(len(a), len(b)) // 2
    if n < 1024:
        return 0.0
    sa = np.abs(np.fft.rfft(a[:n*2]))
    sb = np.abs(np.fft.rfft(b[:n*2]))
    # Mel-like log bands
    edges = np.logspace(np.log10(1), np.log10(len(sa)), n_bands + 1).astype(int)
    ba, bb = np.zeros(n_bands), np.zeros(n_bands)
    for i in range(n_bands):
        lo, hi = edges[i], edges[i+1]
        if lo >= hi: hi = lo + 1
        ba[i] = np.log1p(sa[lo:hi].mean())
        bb[i] = np.log1p(sb[lo:hi].mean())
    return _cos_flat(ba, bb)

# metadata helpers

def write_request(path, caption, lyrics, bpm, duration, seed, cfg):
    import json
    req = {
        "caption": caption,
        "lyrics": lyrics,
        "bpm": bpm,
        "duration": duration,
        "seed": seed,
        "inference_steps": cfg["steps"],
        "guidance_scale": cfg["guidance"],
        "shift": cfg["shift"],
        "vocal_language": "en",
        "thinking": False,
    }
    with open(path, "w") as f:
        json.dump(req, f, indent=4)

# GGML runner

def run_ggml(dump_dir, caption, lyrics, bpm, duration, seed, cfg, noise_file=None):
    if not os.path.isfile(GGML_BIN):
        print(f"[GGML] binary not found: {GGML_BIN}")
        return False
    os.makedirs(dump_dir, exist_ok=True)
    request_json = os.path.join(dump_dir, "request.json")
    write_request(request_json, caption, lyrics, bpm, duration, seed, cfg)
    cmd = [
        GGML_BIN,
        "--dit", cfg["ckpt_dir"], "--vae", VAE_DIR, "--text-encoder", QWEN_DIR,
        "--request", request_json,
        "--dump", dump_dir,
        "--output", os.path.join(dump_dir, "output.wav"),
    ]
    if noise_file:
        cmd += ["--noise-file", noise_file]
    print(f"[GGML] Running {cfg['config_path']}...")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    n = len([f for f in os.listdir(dump_dir) if f.endswith(".bin")])
    if r.returncode != 0:
        if n > 0:
            print(f"[GGML] WARNING: exit {r.returncode} but {n} dump files exist, continuing")
        else:
            print(f"[GGML] FAILED (exit {r.returncode})")
            if r.stdout:
                print(r.stdout[-500:])
            return False
    print(f"[GGML] Done, {n} dump files")
    return True

# Python runner

def run_python(dump_dir, caption, lyrics, bpm, duration, seed, cfg):
    sys.path.insert(0, os.path.join(ROOT, "..", "ACE-Step-1.5"))
    from acestep.handler import AceStepHandler

    os.makedirs(dump_dir, exist_ok=True)
    has_cfg = cfg["guidance"] > 0

    print(f"[Python] Initializing {cfg['config_path']}...")
    handler = AceStepHandler()
    handler.initialize_service(
        project_root=os.path.join(ROOT, "checkpoints"),
        config_path=cfg["config_path"],
        device="cuda",
    )
    model = handler.model

    _dumps = {}

    # Hook 1: text encoder
    orig_text = handler.infer_text_embeddings
    def hooked_text(*a, **kw):
        r = orig_text(*a, **kw)
        _dumps["text_hidden"] = r[0].clone()
        return r
    handler.infer_text_embeddings = hooked_text

    # Hook 2: lyric embeddings
    orig_lyric = handler.infer_lyric_embeddings
    def hooked_lyric(*a, **kw):
        r = orig_lyric(*a, **kw)
        _dumps["lyric_embed"] = r[0].clone()
        return r
    handler.infer_lyric_embeddings = hooked_lyric

    # Hook 3: prepare_condition -> enc_hidden, context
    orig_cond = model.prepare_condition
    def hooked_prepare(*a, **kw):
        r = orig_cond(*a, **kw)
        enc_hs, enc_mask, ctx = r
        _dumps["enc_hidden"] = enc_hs[0].clone()
        _dumps["context"] = ctx[0].clone()
        if has_cfg:
            null_expanded = model.null_condition_emb.expand_as(enc_hs)
            _dumps["null_enc_hidden"] = null_expanded[0].clone()
        return r
    model.prepare_condition = hooked_prepare

    # Hook 4: noise
    orig_noise = model.prepare_noise
    def hooked_noise(*a, **kw):
        n = orig_noise(*a, **kw)
        _dumps["noise"] = n[0].clone()
        return n
    model.prepare_noise = hooked_noise

    # Hook 5: decoder.forward -> per-step vt and xt
    decoder = model.decoder
    _step = [0]
    orig_fwd = decoder.forward
    def hooked_fwd(*args, **kwargs):
        xt_in = args[0] if args else kwargs.get('hidden_states')
        step = _step[0]
        if step > 0 and xt_in is not None:
            _dumps[f"dit_step{step - 1}_xt"] = xt_in[0].clone()
        out = orig_fwd(*args, **kwargs)
        vt = out[0]
        if has_cfg and vt.shape[0] == 2:
            _dumps[f"dit_step{step}_vt_cond"] = vt[0].clone()
            _dumps[f"dit_step{step}_vt_uncond"] = vt[1].clone()
        else:
            _dumps[f"dit_step{step}_vt_cond"] = vt[0].clone()
        if not has_cfg:
            _dumps[f"dit_step{step}_vt"] = vt[0].clone()
        _step[0] += 1
        return out
    decoder.forward = hooked_fwd

    # Hook 5b (SFT only): APG post-guidance vt
    if has_cfg:
        gen_globals = model.generate_audio.__func__.__globals__
        _apg_step = [0]
        orig_apg = gen_globals['apg_forward']
        def hooked_apg(*args, **kwargs):
            result = orig_apg(*args, **kwargs)
            _dumps[f"dit_step{_apg_step[0]}_vt"] = result[0].clone()
            _apg_step[0] += 1
            return result
        gen_globals['apg_forward'] = hooked_apg
        _dumps["null_condition_emb"] = model.null_condition_emb.squeeze().clone()

    # Hook 6: layer intermediates at step 0
    _hooks = []

    def make_hook(name, step_filter=0):
        def hook(module, input, output):
            if _step[0] == step_filter:
                out = output[0] if isinstance(output, tuple) else output
                _dumps[name] = out[0].clone().float()
        return hook

    _hooks.append(decoder.proj_in.register_forward_hook(make_hook("hidden_after_proj_in")))
    _hooks.append(decoder.condition_embedder.register_forward_hook(make_hook("enc_after_cond_emb")))
    _hooks.append(decoder.layers[0].register_forward_hook(make_hook("hidden_after_layer0")))
    _hooks.append(decoder.layers[0].self_attn.register_forward_hook(make_hook("layer0_sa_output")))
    for li in [6, 12, 18, 23]:
        if li < len(decoder.layers):
            _hooks.append(decoder.layers[li].register_forward_hook(make_hook(f"hidden_after_layer{li}")))
    _hooks.append(decoder.time_embed.register_forward_hook(make_hook("temb_t")))

    gen_kwargs = dict(
        captions=caption, lyrics=lyrics, bpm=bpm,
        audio_duration=float(duration), seed=str(seed),
        use_random_seed=False, batch_size=1,
        inference_steps=cfg["steps"], shift=cfg["shift"],
        infer_method="ode", vocal_language="en",
        # Match GGML instruction (production always uses "Generate...")
        instruction="Generate audio semantic tokens based on the given conditions:",
    )

    # "Generate..." instruction triggers is_cover=True in Python even without codes,
    # which round-trips silence through FSQ and changes src_latents.
    # GGML uses raw silence_latent.bin directly. Patch to match.
    orig_build = handler._build_chunk_masks_and_src_latents
    def _patched_build(*a, **kw):
        import torch
        chunk_masks, spans, is_covers, src_latents = orig_build(*a, **kw)
        return chunk_masks, spans, torch.zeros_like(is_covers), src_latents
    handler._build_chunk_masks_and_src_latents = _patched_build
    if has_cfg:
        gen_kwargs["guidance_scale"] = cfg["guidance"]

    tag = f"{cfg['config_path']}, {cfg['steps']} steps"
    if has_cfg:
        tag += f", CFG {cfg['guidance']}"
    print(f"[Python] Generating ({tag})...")
    result = handler.generate_music(**gen_kwargs)

    if not result.get("success"):
        print(f"[Python] Generation failed: {result.get('error', 'unknown')}")
        return False

    for h in _hooks:
        h.remove()

    extra = result.get("extra_outputs", {})
    if extra.get("pred_latents") is not None:
        _dumps["dit_x0"] = extra["pred_latents"][0]

    audios = result.get("audios", [])
    if audios and "tensor" in audios[0]:
        _dumps["vae_audio"] = audios[0]["tensor"].squeeze(0)
        # Save Python WAV for listening comparison
        audio_np = audios[0]["tensor"].squeeze(0).cpu().numpy()  # [2, N]
        wav_path = os.path.join(dump_dir, "output.wav")
        import wave
        n_samples = audio_np.shape[1]
        interleaved = np.empty(2 * n_samples, dtype=np.float32)
        interleaved[0::2] = audio_np[0]
        interleaved[1::2] = audio_np[1]
        pcm = (np.clip(interleaved, -1, 1) * 32767).astype(np.int16)
        with wave.open(wav_path, 'w') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm.tobytes())
        print(f"[Python] Wrote {wav_path}: {n_samples} samples ({n_samples/48000:.2f}s @ 48kHz stereo)")

    for name, tensor in sorted(_dumps.items()):
        save_dump(os.path.join(dump_dir, f"{name}.bin"), tensor)

    print(f"[Python] Done, {len(_dumps)} dump files")
    return True

# comparison

def build_stages(cfg):
    """Build stage list dynamically based on mode config."""
    has_cfg = cfg["guidance"] > 0
    steps = cfg["steps"]

    stages = [
        "text_hidden", "lyric_embed", "enc_hidden", "context", "noise",
        "temb_t", "hidden_after_proj_in", "enc_after_cond_emb",
        "layer0_sa_output", "hidden_after_layer0",
        "hidden_after_layer6", "hidden_after_layer12", "hidden_after_layer18", "hidden_after_layer23",
    ]

    if has_cfg:
        stages += ["null_condition_emb", "null_enc_hidden"]

    # Step dumps: all steps for turbo, sampled for SFT
    if steps <= 8:
        step_indices = list(range(steps))
    else:
        step_indices = list(range(0, steps, 5))
        if (steps - 1) not in step_indices:
            step_indices.append(steps - 1)

    for si in step_indices:
        if has_cfg:
            stages.append(f"dit_step{si}_vt_cond")
            if si < 2:
                stages.append(f"dit_step{si}_vt_uncond")
        stages.append(f"dit_step{si}_vt")
        if si < steps - 1:
            stages.append(f"dit_step{si}_xt")

    stages += ["dit_x0", "vae_audio"]
    return stages

def compare(dirs, stages, tag):
    labels = sorted(dirs.keys())
    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i+1, len(labels))]

    print(f"[{tag}] Cosine similarities GGML vs Python")
    print(f"  {'stage':30s}", end="")
    for a, b in pairs:
        print(f" {a+' vs '+b:>14s}", end="")
    print()

    for stage in stages:
        data = {}
        for label, d in dirs.items():
            f = os.path.join(d, stage + ".bin")
            if os.path.isfile(f):
                data[label] = load_dump(f)

        if not data:
            continue

        print(f"  {stage:30s}", end="")
        for a, b in pairs:
            if a in data and b in data:
                da, sa = data[a]
                db, sb = data[b]
                c = cos(da, db, sa, sb)
                print(f" {c:>14.6f}", end="")
            else:
                print(f" {'N/A':>14s}", end="")
        print()

    # VAE audio spectral comparison
    vae_data = {}
    for label, d in dirs.items():
        f = os.path.join(d, "vae_audio.bin")
        if os.path.isfile(f):
            vae_data[label] = load_dump(f)
    if len(vae_data) >= 2:
        print(f"  {'vae_audio (log spectral)':30s}", end="")
        for a, b in pairs:
            if a in vae_data and b in vae_data:
                c = log_spectral_cos(vae_data[a][0], vae_data[b][0])
                print(f" {c:>14.6f}", end="")
            else:
                print(f" {'N/A':>14s}", end="")
        print()

    # Error growth analysis on xt
    if len(pairs) > 0:
        a_label, b_label = pairs[0]
        a_dir, b_dir = dirs[a_label], dirs[b_label]
        xt_stages = [s for s in stages if "_xt" in s]
        if xt_stages:
            print(f"[{tag}] Error growth GGML vs Python")
            print(f"  {'stage':22s} {'cos':>10s} {'max_err':>10s} {'mean_err':>10s}"
                  f" {'mean_A':>10s} {'std_A':>10s} {'mean_B':>10s} {'std_B':>10s}")
            for stage in xt_stages:
                fa = os.path.join(a_dir, stage + ".bin")
                fb = os.path.join(b_dir, stage + ".bin")
                if os.path.isfile(fa) and os.path.isfile(fb):
                    da_raw, sa = load_dump(fa)
                    db_raw, sb = load_dump(fb)
                    if len(sa) == 2 and len(sb) == 2 and sa[0] == sb[0] and sa[1] == sb[1]:
                        da = da_raw.reshape(sa)
                        db = db_raw.reshape(sb)
                        c_flat = _cos_flat(da.flatten(), db.flatten())
                        c_trans = _cos_flat(da.T.flatten(), db.flatten())
                        if c_trans > c_flat:
                            da = da.T
                        da, db = da.flatten(), db.flatten()
                    else:
                        da, db = da_raw, db_raw
                    n = min(len(da), len(db))
                    da, db = da[:n], db[:n]
                    c = _cos_flat(da, db)
                    diff = np.abs(da - db)
                    print(f"  {stage:22s} {c:10.6f} {diff.max():10.6f} {diff.mean():10.6f}"
                          f" {da.mean():10.6f} {da.std():10.6f} {db.mean():10.6f} {db.std():10.6f}")
                else:
                    missing = []
                    if not os.path.isfile(fa): missing.append(a_label)
                    if not os.path.isfile(fb): missing.append(b_label)
                    print(f"  {stage:22s} missing: {', '.join(missing)}")

# main

def run_mode(mode_name, cfg, args, noise_file):
    """Run a single mode (turbo or sft). Returns True if comparison succeeded."""
    dump_ggml   = os.path.join(SCRIPT_DIR, f"ggml-{mode_name}")
    dump_python = os.path.join(SCRIPT_DIR, f"python-{mode_name}")
    dirs = {}

    tag = mode_name.upper() if mode_name == "sft" else mode_name.capitalize()
    cfg_str = f"steps={cfg['steps']}, shift={cfg['shift']}"
    if cfg['guidance'] > 0:
        cfg_str += f", CFG={cfg['guidance']}"
    print(f"[{tag}] {cfg_str}")

    if not args.skip_ggml:
        if os.path.isdir(dump_ggml):
            shutil.rmtree(dump_ggml)
        if run_ggml(dump_ggml, args.caption, args.lyrics,
                    args.bpm, args.duration, args.seed, cfg,
                    noise_file=noise_file):
            dirs["GGML"] = dump_ggml

    if not args.skip_python:
        if os.path.isdir(dump_python):
            shutil.rmtree(dump_python)
        if run_python(dump_python, args.caption, args.lyrics, args.bpm,
                      args.duration, args.seed, cfg):
            dirs["Python"] = dump_python

    if len(dirs) < 2:
        print(f"[{tag}] Need both backends, got {len(dirs)}: {list(dirs.keys())}")
        return False

    stages = build_stages(cfg)
    compare(dirs, stages, tag)
    return True

def main():
    ap = argparse.ArgumentParser(description="GGML vs Python cosine similarity comparison")
    ap.add_argument("--mode",        default="both", choices=["turbo", "sft", "both"],
                    help="which model(s) to test (default: both)")
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

    # Noise file
    noise_file = args.noise_file
    T_noise = args.duration * 25  # must match Python's torch.randn([1, 64, T])
    if noise_file is None and args.seed == 42:
        auto = os.path.join(SCRIPT_DIR, "rng_philox_seed42.bf16")
        print(f"[Noise] Generating T={T_noise} (duration={args.duration}s)...")
        if not generate_philox_noise(auto, seed=42, T=T_noise):
            print("[Noise] WARNING: cannot generate, GGML will use mt19937")
        if os.path.isfile(auto):
            noise_file = auto

    # Run selected modes
    modes = ["turbo", "sft"] if args.mode == "both" else [args.mode]
    ok = True
    for m in modes:
        if not run_mode(m, MODE_CONFIG[m], args, noise_file):
            ok = False

    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
