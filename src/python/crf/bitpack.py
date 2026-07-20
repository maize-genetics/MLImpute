"""
Bit-packing for the binary founder-match matrix.

The core feature matrix is binary `[positions, K]` (founder matches). Stored as
int8 it is K bytes/site; packed to bits it is ceil(K/8) bytes/site — 8x smaller IO
and memory (K=24 -> 3 bytes vs 24). Per-site founder counts (the IBD-tie statistic
in eval_diploid_ties) are then a popcount over the packed bytes instead of a sum
over K int8s. Labels (H1,H2) stay int8 alongside the packed block.

  pack_matches(M)    [N,T,K] {0,1} int8 -> packed uint8 [N,T,ceil(K/8)] (+ K)
  unpack_matches(P,K)                    -> [N,T,K] int8   (round-trips exactly)
  match_counts(P,K)  packed -> [N,T] int  (#matching founders, via popcount)

These keep the on-disk/whole-genome representation small; the model still consumes
the unpacked float matrix (cheap on GPU), and binary_cells (train_crf) avoids the
per-cell MLP regardless.
"""

import numpy as np


def pack_matches(M):
    """M [..., K] in {0,1} -> packed uint8 [..., ceil(K/8)] along last axis."""
    M = np.asarray(M, dtype=np.uint8)
    return np.packbits(M, axis=-1)                       # bit-order big-endian, K padded


def unpack_matches(P, K):
    """Inverse of pack_matches: packed uint8 [..., nbytes] -> int8 [..., K]."""
    U = np.unpackbits(np.asarray(P, dtype=np.uint8), axis=-1)
    return U[..., :K].astype(np.int8)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)


def match_counts(P, K=None):
    """#matching founders per site from packed bytes via a popcount LUT.
    K is unused (padding bits are zero) but accepted for signature symmetry."""
    return _POPCOUNT[np.asarray(P, dtype=np.uint8)].sum(axis=-1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M = (rng.random((3, 1000, 24)) < 0.3).astype(np.int8)
    P = pack_matches(M)
    U = unpack_matches(P, 24)
    assert np.array_equal(U, M), "round-trip failed"
    assert np.array_equal(match_counts(P), M.sum(-1)), "popcount mismatch"
    print(f"round-trip OK; int8 {M.nbytes:,}B -> packed {P.nbytes:,}B "
          f"({M.nbytes / P.nbytes:.1f}x smaller); popcount matches sum")
