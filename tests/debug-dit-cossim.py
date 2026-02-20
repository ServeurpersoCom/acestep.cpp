#!/usr/bin/env python3
"""GGML vs Python cosine similarity comparison for ACE-Step DiT.

Run from tests/ directory. All paths relative to CWD.

Usage:
    cd tests/
    ./debug-dit-cossim.py                     # turbo BF16
    ./debug-dit-cossim.py --quant Q6_K        # turbo Q6_K
    ./debug-dit-cossim.py --mode sft           # SFT BF16
"""
import os, sys, subprocess, struct, shutil, argparse, json
import numpy as np

SEED = 42

MODE_CONFIG = {
    "turbo": {
        "gguf_base": "acestep-v15-turbo",
        "config_path": "acestep-v15-turbo",
        "steps": 8, "shift": 3.0, "guidance": 0.0,
    },
    "sft": {
        "gguf_base": "acestep-v15-sft",
        "config_path": "acestep-v15-sft",
        "steps": 50, "shift": 1.0, "guidance": 7.0,
    },
}

def load_request():
    if not os.path.isfile("request0.json"):
        print("[Error] request0.json not found in CWD")
        sys.exit(1)
    with open("request0.json") as f:
        req = json.load(f)
    print(f"[Request] Loaded request0.json")
    return req

def generate_philox_noise(path, T, C=64):
    try:
        import torch
    except ImportError:
        print("[Noise] ERROR: PyTorch required")
        return False
    if not torch.cuda.is_available():
        print("[Noise] ERROR: CUDA required for Philox RNG")
        return False
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    noise = torch.randn([1, C, T], generator=gen, device="cuda", dtype=torch.bfloat16)
    raw = noise[0].cpu().contiguous().view(torch.uint8).numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)
    print(f"[Noise] Generated {path}: [{T}, {C}] bf16 ({len(raw)} bytes)")
    return True

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
    if shape_a and shape_b and len(shape_a) == 2 and len(shape_b) == 2:
        if shape_a[0] == shape_b[1] and shape_a[1] == shape_b[0]:
            ra = a.reshape(shape_a)
            rb = b.reshape(shape_b)
            c_normal = _cos_flat(ra.flatten(), rb.flatten())
            c_transposed = _cos_flat(ra.T.flatten(), rb.flatten())
            if c_transposed > c_normal:
                return c_transposed
            return c_normal
    return _cos_flat(a, b)

def log_spectral_cos(a, b, sr=48000, n_bands=128):
    n = min(len(a), len(b))
    sa = np.abs(np.fft.rfft(a[:n*2]))
    sb = np.abs(np.fft.rfft(b[:n*2]))
    edges = np.logspace(np.log10(1), np.log10(len(sa)), n_bands + 1).astype(int)
    ba, bb = np.zeros(n_bands), np.zeros(n_bands)
    for i in range(n_bands):
        lo, hi = edges[i], edges[i+1]
        if lo >= hi: hi = lo + 1
        ba[i] = np.log1p(sa[lo:hi].mean())
        bb[i] = np.log1p(sb[lo:hi].mean())
    return _cos_flat(ba, bb)

def codes_to_python_format(codes_csv):
    """Convert '43316,18426,...' to '<|audio_code_43316|><|audio_code_18426|>...'"""
    if not codes_csv:
        return ""
    return "".join(f"<|audio_code_{c.strip()}|>" for c in codes_csv.split(",") if c.strip())

# GGML runner

def run_ggml(dump_dir, req, cfg, gguf_path, noise_file):
    ggml_bin = "../build/dit-vae"
    if not os.path.isfile(ggml_bin):
        print(f"[GGML] binary not found: {ggml_bin}")
        return False
    os.makedirs(dump_dir, exist_ok=True)

    # Build request from input, override mode-specific params
    merged = dict(req)
    merged["seed"] = SEED
    merged["inference_steps"] = cfg["steps"]
    merged["guidance_scale"] = cfg["guidance"]
    merged["shift"] = cfg["shift"]
    merged["thinking"] = False
    request_json = os.path.join(dump_dir, "request0.json")
    with open(request_json, "w") as f:
        json.dump(merged, f, indent=4)

    cmd = [
        ggml_bin,
        "--dit", gguf_path,
        "--text-encoder", "../models/Qwen3-Embedding-0.6B-BF16.gguf",
        "--vae", "../models/vae-BF16.gguf",
        "--request", request_json,
        "--dump", dump_dir,
        "--output", os.path.join(dump_dir, "output.wav"),
    ]
    if noise_file:
        cmd += ["--noise-file", noise_file]
    print(f"[GGML] Running {os.path.basename(gguf_path)}...")
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

def run_python(dump_dir, req, cfg):
    sys.path.insert(0, "../../ACE-Step-1.5")
    from acestep.handler import AceStepHandler

    os.makedirs(dump_dir, exist_ok=True)
    has_cfg = cfg["guidance"] > 0

    caption  = req["caption"]
    lyrics   = req.get("lyrics", "")
    bpm      = req.get("bpm", 0)
    duration = req["duration"]
    language = req.get("vocal_language", "en")

    print(f"[Python] Initializing {cfg['config_path']}...")
    handler = AceStepHandler()
    handler.initialize_service(
        project_root="..",
        config_path=cfg["config_path"],
        device="cuda",
    )
    model = handler.model
    _dumps = {}

    orig_text = handler.infer_text_embeddings
    def hooked_text(*a, **kw):
        r = orig_text(*a, **kw)
        _dumps["text_hidden"] = r[0].clone()
        return r
    handler.infer_text_embeddings = hooked_text

    orig_lyric = handler.infer_lyric_embeddings
    def hooked_lyric(*a, **kw):
        r = orig_lyric(*a, **kw)
        _dumps["lyric_embed"] = r[0].clone()
        return r
    handler.infer_lyric_embeddings = hooked_lyric

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

    orig_noise = model.prepare_noise
    def hooked_noise(*a, **kw):
        n = orig_noise(*a, **kw)
        _dumps["noise"] = n[0].clone()
        return n
    model.prepare_noise = hooked_noise

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
        _hooks.append(decoder.layers[li].register_forward_hook(make_hook(f"hidden_after_layer{li}")))

    _hooks.append(decoder.time_embed.register_forward_hook(make_hook("temb_t")))

    # Hook detokenizer (runs during prepare_condition, before diffusion)
    if hasattr(model, 'detokenizer'):
        def detok_hook(module, input, output):
            _dumps["detok_output"] = output[0].clone().float()
        _hooks.append(model.detokenizer.register_forward_hook(detok_hook))

    gen_kwargs = dict(
        captions=caption, lyrics=lyrics, bpm=bpm,
        audio_duration=float(duration), seed=str(SEED),
        use_random_seed=False, batch_size=1,
        inference_steps=cfg["steps"], shift=cfg["shift"],
        guidance_scale=cfg["guidance"],
        infer_method="ode", vocal_language=language,
        audio_code_string=codes_to_python_format(req.get("audio_codes", "")),
        key_scale=req.get("keyscale", ""),
        time_signature=req.get("timesignature", ""),
    )

    # When audio_codes are present, Python auto-sets is_covers=True via
    # conditioning_masks.py (instruction match + has_code_hint).
    # This makes it use decoded codes as context, matching C++ behavior.
    # Do NOT patch is_covers to False, that would use silence instead of codes.

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
        audio_np = audios[0]["tensor"].squeeze(0).cpu().numpy()
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
    has_cfg = cfg["guidance"] > 0
    steps = cfg["steps"]
    stages = [
        "text_hidden", "lyric_embed", "enc_hidden", "detok_output", "context", "noise",
        "temb_t", "hidden_after_proj_in", "enc_after_cond_emb",
        "layer0_sa_output", "hidden_after_layer0",
        "hidden_after_layer6", "hidden_after_layer12", "hidden_after_layer18", "hidden_after_layer23",
    ]
    if has_cfg:
        stages += ["null_condition_emb", "null_enc_hidden"]
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

def run_mode(mode_name, cfg, req, gguf_path, noise_file):
    dump_ggml   = f"ggml-{mode_name}"
    dump_python = f"python-{mode_name}"

    tag = mode_name.upper() if mode_name == "sft" else mode_name.capitalize()
    cfg_str = f"steps={cfg['steps']}, shift={cfg['shift']}"
    if cfg['guidance'] > 0:
        cfg_str += f", CFG={cfg['guidance']}"
    print(f"[{tag}] {cfg_str} | {os.path.basename(gguf_path)}")

    if os.path.isdir(dump_ggml):
        shutil.rmtree(dump_ggml)
    if not run_ggml(dump_ggml, req, cfg, gguf_path, noise_file):
        print(f"[{tag}] GGML failed")
        return False

    if os.path.isdir(dump_python):
        shutil.rmtree(dump_python)
    if not run_python(dump_python, req, cfg):
        print(f"[{tag}] Python failed")
        return False

    stages = build_stages(cfg)
    compare({"GGML": dump_ggml, "Python": dump_python}, stages, tag)
    return True

def main():
    ap = argparse.ArgumentParser(description="GGML vs Python cosine similarity comparison")
    ap.add_argument("--mode", default="turbo", choices=["turbo", "sft", "both"],
                    help="which model(s) to test (default: turbo)")
    ap.add_argument("--quant", default="BF16",
                    help="quantization suffix for GGUF (default: BF16, e.g. Q6_K, Q8_0)")
    args = ap.parse_args()

    req = load_request()
    duration = req["duration"]

    # Noise
    T_noise = int(duration) * 25
    noise_file = "rng_philox_seed42.bf16"
    print(f"[Noise] Generating T={T_noise} (duration={duration}s)...")
    if not generate_philox_noise(noise_file, T=T_noise):
        print("[Noise] WARNING: cannot generate, GGML will use mt19937")
        noise_file = None

    modes = ["turbo", "sft"] if args.mode == "both" else [args.mode]
    ok = True
    for m in modes:
        cfg = MODE_CONFIG[m]
        gguf_path = f"../models/{cfg['gguf_base']}-{args.quant}.gguf"
        if not os.path.isfile(gguf_path):
            print(f"[Error] GGUF not found: {gguf_path}")
            ok = False
            continue
        if not run_mode(m, cfg, req, gguf_path, noise_file):
            ok = False

    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
