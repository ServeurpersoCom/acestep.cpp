#!/usr/bin/env python3
"""Compare C++ vs Python FSQ tokenizer, code by code.

Runs ace-understand --dump to get C++ VAE latents + FSQ codes,
then runs the Python tokenizer on the same latents and compares.

Run from tests/ directory:
    ./debug-tok-cossim.py                  # turbo, 1s sine
    ./debug-tok-cossim.py --mode sft       # SFT
    ./debug-tok-cossim.py --duration 5     # 5s test audio
    ./debug-tok-cossim.py --wav input.wav  # custom WAV
"""
import sys, os, subprocess, argparse, struct, shutil, math
import numpy as np

FSQ_LEVELS = [8, 8, 8, 5, 5, 5]

ACE_BIN = "../build/ace-understand"
VAE_GGUF = "../models/vae-BF16.gguf"

MODE_CONFIG = {
    "turbo": {
        "dit_gguf":    "../models/acestep-v15-turbo-BF16.gguf",
        "config_path": "acestep-v15-turbo",
    },
    "sft": {
        "dit_gguf":    "../models/acestep-v15-sft-BF16.gguf",
        "config_path": "acestep-v15-sft",
    },
}


def generate_test_wav(path, duration=1.0, sr=48000):
    """Generate a short stereo WAV (440Hz sine) for testing."""
    ns = int(sr * duration)
    t = np.arange(ns, dtype=np.float64) / sr
    mono = (np.sin(2 * math.pi * 440 * t) * 16000).astype(np.int16)
    nch = 2
    data = np.column_stack([mono, mono]).tobytes()
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(data)))
        f.write(b'WAVEfmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, nch, sr, sr * nch * 2, nch * 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(data)))
        f.write(data)


def load_dump(path):
    """Load debug.h format: [ndim:i32] [shape:i32*ndim] [data:f32*numel]."""
    raw = np.fromfile(path, dtype=np.float32)
    ndim = struct.unpack('i', struct.pack('f', raw[0]))[0]
    shape = [struct.unpack('i', struct.pack('f', raw[1 + i]))[0] for i in range(ndim)]
    data = raw[1 + ndim:]
    return data.reshape(shape)


def fsq_decode_index(index):
    dims = []
    for L in FSQ_LEVELS:
        dims.append(index % L)
        index //= L
    return dims


def run_cpp(wav_path, dit_gguf, dump_dir):
    """Run ace-understand --dump (tok-only, no LM). Stderr goes to terminal."""
    if os.path.isdir(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(dump_dir)
    cmd = [ACE_BIN,
           "--src-audio", wav_path,
           "--dit", dit_gguf,
           "--vae", VAE_GGUF,
           "--dump", dump_dir]
    print("[GGML] Running ace-understand --dump...")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    lat_path = os.path.join(dump_dir, "tok_latents.bin")
    cod_path = os.path.join(dump_dir, "tok_codes.bin")
    if r.returncode != 0 or not os.path.isfile(lat_path):
        print(f"[GGML] FAILED (exit {r.returncode})")
        return None, None
    latents = load_dump(lat_path)
    codes = np.fromfile(cod_path, dtype=np.int32)
    print(f"[GGML] Done, {latents.shape[0]} latent frames -> {len(codes)} codes")
    return latents, codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="turbo", choices=["turbo", "sft"])
    parser.add_argument("--duration", type=float, default=1.0, help="Test audio duration (seconds)")
    parser.add_argument("--wav", type=str, default=None, help="Custom WAV file instead of generated")
    args = parser.parse_args()
    cfg = MODE_CONFIG[args.mode]

    wav_path = args.wav
    if not wav_path:
        wav_path = "tok-test-input.wav"
        generate_test_wav(wav_path, args.duration)
        print(f"[Input] Generated {args.duration:.1f}s 440Hz stereo WAV")

    # Step 1: C++ (ace-understand --dump)
    dump_dir = "tok-dump"
    latents, cpp_codes = run_cpp(wav_path, cfg["dit_gguf"], dump_dir)
    if latents is None:
        return 1

    # Step 2: Python tokenizer on the same latents
    print("[Python] Loading model...")
    import torch
    sys.path.insert(0, '../../ACE-Step-1.5')
    from acestep.handler import AceStepHandler
    from einops import rearrange

    handler = AceStepHandler()
    handler.initialize_service(
        project_root="..",
        config_path=cfg["config_path"],
        device='cpu',
    )
    tokenizer = handler.model.tokenizer.float()

    T_25Hz = latents.shape[0]
    pad = (5 - (T_25Hz % 5)) % 5
    lat_np = latents
    if pad > 0:
        sl_bin = os.path.join("..", "checkpoints", cfg["config_path"], "silence_latent.bin")
        silence = np.fromfile(sl_bin, dtype=np.float32).reshape(-1, 64)
        lat_np = np.concatenate([lat_np, silence[:pad]], axis=0)

    lat_t = torch.tensor(lat_np, dtype=torch.float32).unsqueeze(0)
    x = rearrange(lat_t, 'n (t_patch p) d -> n t_patch p d', p=5)
    with torch.no_grad():
        _, indices = tokenizer(x)
    py_codes = indices.squeeze().cpu().numpy().flatten()
    print(f"[Python] {len(py_codes)} codes")

    # Step 3: Compare
    n = min(len(cpp_codes), len(py_codes))
    matches = sum(1 for i in range(n) if cpp_codes[i] == py_codes[i])
    pct = 100.0 * matches / n if n > 0 else 0
    print(f"[Compare] GGML vs Python ({n} codes)")
    print(f"match: {matches}/{n} ({pct:.1f}%)")

    mismatches = [(i, int(cpp_codes[i]), int(py_codes[i]))
                  for i in range(n) if cpp_codes[i] != py_codes[i]]
    if mismatches:
        off_by_one = 0
        for _, c, p in mismatches:
            cd, pd = fsq_decode_index(c), fsq_decode_index(p)
            diffs = [abs(cd[j] - pd[j]) for j in range(6)]
            if sum(1 for d in diffs if d != 0) == 1 and max(diffs) == 1:
                off_by_one += 1
        print(f"off-by-1 in 1 dim: {off_by_one}/{len(mismatches)}")
        for i, c, p in mismatches[:5]:
            cd, pd = fsq_decode_index(c), fsq_decode_index(p)
            diff_dims = [j for j in range(6) if cd[j] != pd[j]]
            print(f"code[{i}]: GGML={c} Python={p} dims={diff_dims}")

    print(f"[Summary]")
    if pct == 100:
        print(f"PASS: all {n} codes match")
    elif pct >= 80:
        print(f"WARN: {pct:.0f}% match (precision diffs at FSQ boundaries)")
    else:
        print(f"FAIL: {pct:.0f}% match")

    if not args.wav:
        os.remove(wav_path)
    return 0 if pct == 100 else 1


if __name__ == '__main__':
    sys.exit(main())
