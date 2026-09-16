#!/usr/bin/env python3
"""Comparator for test_paged_attn_wmma.cpp two-mode dumps.

Usage: compare_paged_attn.py ref.bin cand.bin [--tol 3e-3]
The dump layout is case order from the test source; the test prints the
per-case shape line. The V_DOT2 reference dequantizes K/V via dp4a-integer
paths while the WMMA kernel dequantizes through a half2 tile, a wider gap
than the contiguous fattn differential. Per-type tolerances covering the
shipped case set: f16 2e-3, q8_0 3e-3, q4_0 6e-3 (the 4-bit lattice is
coarser, so the same dequant-path difference lands ~2x higher; measured
3.4-4.3e-3 max, 3e-4 mean, <2e-4 of elements over 3e-3). This comparator
reports the global max over all concatenated cases, so runs that include
q4_0 cases must raise --tol accordingly; the default keeps the strict
f16/q8_0 bound.
"""
import sys
import numpy as np

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tol = 3e-3
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1])
    ref = np.fromfile(args[0], dtype=np.float32)
    cand = np.fromfile(args[1], dtype=np.float32)
    assert ref.shape == cand.shape, f"shape mismatch: {ref.shape} vs {cand.shape}"
    diff = np.abs(ref - cand)
    maxd = float(diff.max())
    # per-case comparison would need shape metadata; report the global max.
    print(f"elements={ref.size} max_abs_diff={maxd:.6e} tol={tol:.1e} -> {'PASS' if maxd < tol else 'FAIL'}")

if __name__ == "__main__":
    main()
