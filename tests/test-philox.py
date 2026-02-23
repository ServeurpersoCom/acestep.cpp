#!/usr/bin/env python3
"""Verify C++ Philox matches PyTorch CUDA torch.randn(dtype=bf16).

Run from tests/ directory:
    ./test-philox.py
"""
import subprocess, sys, os, random
import numpy as np

COUNT = 64 * 25 * 120  # 64ch * 25Hz * 120s = 192000 (2 minutes, max duration)

def build():
    if not os.path.isfile("test-philox.cpp"):
        print("ERROR: test-philox.cpp not found (run from tests/ directory)")
        sys.exit(1)
    r = subprocess.run(["make", "test-philox"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Build failed:\n{r.stderr}")
        sys.exit(1)

def compare(seed):
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    subprocess.run(["./test-philox", str(seed), str(COUNT), "philox-noise.f32"],
                   capture_output=True)
    cpp = np.fromfile("philox-noise.f32", dtype=np.float32)

    gen = torch.Generator(device="cuda").manual_seed(seed)
    py = torch.randn([COUNT], generator=gen, device="cuda", dtype=torch.bfloat16)
    py = py.float().cpu().numpy()

    n = min(len(cpp), len(py))
    exact = int(np.sum(cpp[:n] == py[:n]))
    diff = np.abs(cpp[:n] - py[:n])
    d = np.linalg.norm(cpp[:n]) * np.linalg.norm(py[:n])
    cos = float(np.dot(cpp[:n], py[:n]) / d) if d > 0 else 0.0
    diffs = n - exact

    status = "PERFECT" if exact == n else "OK" if cos > 0.9999 else "FAIL"
    print(f"  seed={seed:<12d}  {exact}/{n} ({100*exact/n:.2f}%)  "
          f"cos={cos:.8f}  max_diff={diff.max():.8f}  diffs={diffs}  {status}")
    return status != "FAIL"

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build()

    print(f"Philox4x32-10 vs PyTorch CUDA bf16 | {COUNT} floats "
          f"({COUNT//64}frames, {COUNT//64//25}s @ 25Hz, 64ch)")
    print(f"Press Enter for random seed, type a number for specific seed, q to quit.\n")

    ok = True
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in ("q", "quit", "exit"):
            break
        seed = int(line) if line.lstrip('-').isdigit() else random.randint(0, 2**63 - 1)
        if not compare(seed):
            ok = False

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
