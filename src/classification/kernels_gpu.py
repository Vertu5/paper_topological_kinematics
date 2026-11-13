"""
kernels_gpu.py - Triton kernels of the fusion and of affinity propagation
==========================================================================
Each kernel repeats, on the GPU, the float64 arithmetic of the CPU code in the same order, so that
both give the same numbers bit for bit. Two rules make this possible:
  * no fused multiply-add: every kernel is compiled with enable_fp_fusion=False, so a * b + c is
    rounded twice, as numpy and scipy round it;
  * a sum runs in a fixed order: the terms of a local-kernel product in increasing column order (the
    order of scipy's CSR product), the terms of a column sum row after row (the order of numpy's sum
    along axis 0).

Author: Olivier Vertu Ndinga Oba
"""
import torch
import triton
import triton.language as tl

NOFMA = dict(enable_fp_fusion=False)


@triton.jit
def _local_product(X, COLS, W, OUT, n, k, BLOCK: tl.constexpr):
    """OUT[i, :] = sum over j = 0..k-1, in this order, of W[i, j] * X[COLS[i, j], :]."""
    i = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.zeros([BLOCK], dtype=tl.float64)
    for j in range(k):
        c = tl.load(COLS + i * k + j)
        w = tl.load(W + i * k + j)
        acc += w * tl.load(X + c * n + offs, mask=mask, other=0.0)
    tl.store(OUT + i * n + offs, acc, mask=mask)


def local_product(X, cols, w, out=None, block=256):
    """Product K @ X of a local kernel K (row i holds w[i, j] at column cols[i, j]) with a dense X,
    equal bit for bit to scipy's CSR product when cols is sorted in each row."""
    n, k = cols.shape
    out = torch.empty_like(X) if out is None else out
    _local_product[(n, triton.cdiv(X.shape[1], block))](X, cols, w, out, X.shape[1], k, BLOCK=block, **NOFMA)
    return out


@triton.jit
def _ap_responsibility(S, A, R, DAMP, n, BLOCK: tl.constexpr):
    """One row of the responsibility update of scikit-learn's affinity propagation (float64)."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    base = row * n
    s = tl.load(S + base + offs, mask=mask, other=0.0)
    a = tl.load(A + base + offs, mask=mask, other=0.0)
    t = tl.where(mask, a + s, float("-inf"))
    y = tl.max(t, axis=0)
    first = tl.argmax(t, axis=0, tie_break_left=True)
    y2 = tl.max(tl.where(offs == first, float("-inf"), t), axis=0)
    d = tl.load(DAMP)
    c = tl.load(DAMP + 1)                       # 1 - damping, rounded as in Python
    new = (s - tl.where(offs == first, y2, y)) * c
    r = tl.load(R + base + offs, mask=mask, other=0.0)
    tl.store(R + base + offs, r * d + new, mask=mask)


@triton.jit
def _ap_column_sums(R, CS, n, BLOCK: tl.constexpr):
    """CS[b, k] = sum over i of max(R[b, i, k], 0), with R[b, k, k] kept as it is, added row after row."""
    b = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.zeros([BLOCK], dtype=tl.float64)
    base = b * n * n
    for i in range(n):
        r = tl.load(R + base + i * n + offs, mask=mask, other=0.0)
        acc += tl.where(offs == i, r, tl.maximum(r, 0.0))
    tl.store(CS + b * n + offs, acc, mask=mask)


@triton.jit
def _ap_availability(R, A, CS, DAMP, n, BLOCK: tl.constexpr):
    """One row of the availability update of scikit-learn's affinity propagation (float64)."""
    row = tl.program_id(0).to(tl.int64)
    b = row // n
    i = row % n
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    base = row * n
    r = tl.load(R + base + offs, mask=mask, other=0.0)
    cs = tl.load(CS + b * n + offs, mask=mask, other=0.0)
    diag = offs == i
    t = tl.where(diag, r, tl.maximum(r, 0.0)) - cs
    t = tl.where(diag, t, tl.maximum(t, 0.0)) * tl.load(DAMP + 1)
    a = tl.load(A + base + offs, mask=mask, other=0.0)
    tl.store(A + base + offs, a * tl.load(DAMP) - t, mask=mask)


def ap_iteration(S, A, R, CS, damp):
    """One iteration of affinity propagation on a batch (B, n, n), messages updated in place.
    `damp` is a float64 tensor (damping, 1 - damping)."""
    B, n, _ = S.shape
    block = triton.next_power_of_2(n)
    warps = 4 if block <= 1024 else (8 if block <= 4096 else 16)
    _ap_responsibility[(B * n,)](S, A, R, damp, n, BLOCK=block, num_warps=warps, **NOFMA)
    cb = 32
    _ap_column_sums[(B, triton.cdiv(n, cb))](R, CS, n, BLOCK=cb, **NOFMA)
    ab = 256
    _ap_availability[(B * n, triton.cdiv(n, ab))](R, A, CS, damp, n, BLOCK=ab, **NOFMA)
