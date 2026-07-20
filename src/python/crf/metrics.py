"""
Path metrics shared by the CRF and HMM evaluators.

A *breakpoint* is a site t (1..T-1) where the founder path switches
(x[t] != x[t-1]). Breakpoint precision/recall compares predicted switch
positions to the true ones, with a position tolerance (a predicted breakpoint
counts if a true one lies within ±tol sites, and vice-versa) since exact
single-site localization of a recombination is not expected.
"""

import torch
import torch.nn.functional as F


def _switches(x):
    """x [B,T] long -> [B,T-1] bool, True where the path switches."""
    return x[:, 1:] != x[:, :-1]


def _dilate(mask, tol):
    """Widen each True by ±tol along the last dim (max-pool). mask [B,L] bool."""
    if tol <= 0:
        return mask
    xf = mask.float().unsqueeze(1)                      # [B,1,L]
    d = F.max_pool1d(xf, kernel_size=2 * tol + 1, stride=1, padding=tol)
    return d.squeeze(1) > 0


@torch.no_grad()
def breakpoint_counts(pred, tags, tol=0):
    """
    pred, tags: [B,T] long. Returns dict of summed counts (ints) so callers can
    accumulate across batches and divide once at the end:
      tp_prec : predicted breakpoints within ±tol of a true one
      n_pred  : predicted breakpoints
      tp_rec  : true breakpoints within ±tol of a predicted one
      n_true  : true breakpoints
    precision = tp_prec / n_pred,  recall = tp_rec / n_true.
    """
    pb = _switches(pred)
    tb = _switches(tags)
    tp_prec = (pb & _dilate(tb, tol)).sum().item()
    tp_rec = (tb & _dilate(pb, tol)).sum().item()
    return {"tp_prec": tp_prec, "n_pred": pb.sum().item(),
            "tp_rec": tp_rec, "n_true": tb.sum().item()}


def prf(acc):
    """Turn accumulated counts into (precision, recall, f1)."""
    p = acc["tp_prec"] / max(acc["n_pred"], 1)
    r = acc["tp_rec"] / max(acc["n_true"], 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1
