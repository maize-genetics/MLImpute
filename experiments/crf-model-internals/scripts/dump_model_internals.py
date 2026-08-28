#!/usr/bin/env python
"""
Non-invasive dump of GRITSCRFDiploid internals (encoder hidden state, gate,
recombination cost, per-founder emissions, affinity, true posteriors,
heterozygosity-prior effect) for a set of real simval rows plus matched
sim-training comparison groups. Nothing in the model repo is edited --
every extra tensor is captured with forward hooks or a plain-Python
reimplementation of the (non-jit) `_dcrf_marginal` recursion.

See /home/zrm22/.claude/plans/wondrous-discovering-octopus.md for why:
briefly, per-founder read support turned out NOT to distinguish inbred (which
scores well) from hybrid (which doesn't) once stratified by founder count k
(support_profile.py). This dump is built to discriminate three remaining
hypotheses instead:
  H2 -- path-structure mismatch: training never saw a 0-breakpoint constant
        pair; does the decoded path over-switch on a real hybrid?
  H3 -- het-prior miscalibration: the fixed homo_penalty*homo_scale term
        fights rather than scales with the emission signal.
  H4 -- pair-decode ambiguity: only one gamete's read is observed per site,
        so is the joint pair harder to resolve than the per-founder identity?

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/dump_model_internals.py \
        --rows IDX-INBRED__Oh43__0.1x IDX-INBRED__Il14H__0.1x IDX-HYB__Oh43xIl14H__0.1x \
        --n-windows 24
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

GRITS_REPO = Path("/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness")
CORPUS_SCRIPTS = Path(
    "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/scripts")
sys.path.insert(0, str(GRITS_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
sys.path.insert(0, str(CORPUS_SCRIPTS))  # needed by simval_truth_labels' `import mosaic`

from python.crf.train_diploid import (  # noqa: E402
    GRITSCRFDiploid, _founder_affinity, _het_scale, _dcrf_viterbi)
from refbias_parse import split_individual_name  # noqa: E402
import simval_truth_labels as stl  # noqa: E402

GRITS_WORKDIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
SIMVAL_EVAL = GRITS_WORKDIR / "scratch/simval_eval"
SIM_DATA = GRITS_WORKDIR / "data/training/sim_diploid_512_affinity.npy"
SIM_IND = GRITS_WORKDIR / "data/training/sim_diploid_512_affinity.ind.npy"
DEFAULT_CKPT = (GRITS_WORKDIR / "checkpoints/diploid-affinity-sim512-h3/"
                "d-epoch=04-val_pair_acc=0.6179.ckpt")
OUT_ROOT = GRITS_WORKDIR / "results/model_internals"

K = 24  # real founders (excludes the +1 null/unknown state)
KIND_HOMO_SCALE = {"inbred": 0.0, "hybrid": 1.0, "ril": 0.5, "ril2": 0.0}
DEFAULT_ROWS = [
    "IDX-INBRED__Oh43__0.1x",
    "IDX-INBRED__Il14H__0.1x",
    "IDX-HYB__Oh43xIl14H__0.1x",
]


# --------------------------------------------------------------------------- #
#  Model loading + hooks                                                       #
# --------------------------------------------------------------------------- #

class Capture:
    """Forward-hook sink: model repo code is never edited, we just record
    module outputs as they pass through."""
    def __init__(self):
        self.H = None          # encoder pos_encoder output [B,T,d]
        self.emis_f = None     # encoder final per-founder emission [B,T,K]
        self.ext_bias_out = None  # raw ext_bias(ext_emb) [B,K,1], pre-broadcast
        self.het = None        # per-locus het logit [B,T], learned_het checkpoints only

    def hook_pos_encoder(self, module, inp, out):
        self.H = out.detach()

    def hook_encoder(self, module, inp, out):
        self.emis_f = out[0].detach()
        self.het = out[3].detach() if len(out) > 3 else None   # per-locus het logit, learned_het only

    def hook_ext_bias(self, module, inp, out):
        self.ext_bias_out = out.detach()


def load_model(ckpt_path):
    model = GRITSCRFDiploid.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    cap = Capture()
    model.encoder.pos_encoder.register_forward_hook(cap.hook_pos_encoder)
    model.encoder.register_forward_hook(cap.hook_encoder)
    if model.encoder.ext_bias is not None:
        model.encoder.ext_bias.register_forward_hook(cap.hook_ext_bias)
    return model, cap


# --------------------------------------------------------------------------- #
#  Posterior forward-backward (plain reimplementation of _dcrf_marginal,       #
#  which is NOT jit-scripted -- safe to mirror; kept structurally identical    #
#  so its argmax matches _dcrf_marginal exactly, see plan's Verification.)     #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_posteriors(emis, c, nsw, stay_bonus):
    """emis [B,T,P], c [B,T], nsw/[P,P], stay_bonus scalar -> posterior [B,T,P]
    (softmax-normalized per site; argmax reproduces _dcrf_marginal's pred)."""
    emis = emis.float(); c = c.float(); nsw = nsw.float(); stay_bonus = stay_bonus.float()
    stay_mask = (nsw == 0).float()
    B, T, P = emis.shape
    alpha = torch.empty(B, T, P)
    alpha[:, 0] = emis[:, 0]
    for t in range(1, T):
        tr_t = -c[:, t, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        alpha[:, t] = emis[:, t] + torch.logsumexp(alpha[:, t - 1].unsqueeze(2) + tr_t, dim=1)
    beta = torch.zeros(B, T, P)
    for t in range(T - 2, -1, -1):
        tr_n = -c[:, t + 1, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        msg = emis[:, t + 1] + beta[:, t + 1]
        beta[:, t] = torch.logsumexp(tr_n + msg.unsqueeze(1), dim=2)
    logpost = alpha + beta
    return torch.softmax(logpost, dim=-1)


def pca_top8(H):
    """H [N,d] (already subsampled/flattened) -> (coords [N,8], explained_var [8])."""
    Hc = H - H.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Hc, full_matrices=False)
    n_comp = min(8, S.shape[0])
    coords = (U[:, :n_comp] * S[:n_comp])
    var = (S ** 2) / (H.shape[0] - 1)
    return coords.numpy(), (var[:n_comp] / var.sum()).numpy()


# --------------------------------------------------------------------------- #
#  Per-row internals                                                           #
# --------------------------------------------------------------------------- #

def _pick_window_idx(n_total, n_want):
    if n_total <= n_want:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, n_want).astype(int)


SOURCE_K = 25  # panel size before the fixed founder-drop (matches raw.npy.gametes.tsv)


def _window_row_indices(bins_tsv_path, window_size=512):
    """Reproduces python.crf.ropebwt_npy_to_matrix.py's windowing EXACTLY
    (contiguous window_size row-chunks per contig, in on-disk order, dropping
    a trailing partial window per contig) -- but returns the RAW ROW INDICES
    per window instead of slicing features, so it can be reused to window any
    per-row array (here: per-row truth labels) into the same window axis
    `windowed_k24_fixdrop23.npy` already uses."""
    bins_df = pd.read_csv(bins_tsv_path, sep="\t")
    windows = []
    for _contig, idx in bins_df.groupby("contig", sort=False).indices.items():
        idx = np.sort(idx)
        start = 0
        while start + window_size <= len(idx):
            windows.append(idx[start:start + window_size])
            start += window_size
    return windows


def _window_ril_truth(row_dir, source_names, dropped_idx, dataset_id, individual, parents,
                       window_size=512):
    """Per-window (h1,h2) TRUE founder-index arrays [n_windows, T] for a RIL
    row, in the SAME TARGET (post-drop, 24-founder) index space and SAME
    window ordering as windowed_k24_fixdrop23.npy. Uses
    simval_truth_labels.bin_truth_labels (real breakpoint mosaic, delegates
    to simval_oracle_bed.build_ril_mosaics -- already validated against the
    real corpus truth gVCFs, max 0.0235% error, per last session's memory)
    for per-RAW-ROW truth, then windows it exactly like
    ropebwt_npy_to_matrix.py windows the features, then remaps SOURCE
    (25-founder) indices into TARGET (24-founder, post-drop) space -- the
    same remap ropebwt_npy_to_matrix.py itself applies to the feature labels."""
    bins_path = row_dir / "raw.npy.bins.tsv"
    labels, _ = stl.bin_truth_labels(
        bins_path, kind="ril", gamete_names=source_names,
        dataset_id=dataset_id, individual=individual,
        parent_a=parents[0], parent_b=parents[1])   # [n_rows,2] SOURCE-space, -1 if unresolved

    remap = np.full(SOURCE_K + 1, K, dtype=np.int64)   # dropped/-1/unknown -> TARGET K (unknown)
    keep = [i for i in range(SOURCE_K) if i != dropped_idx]
    for target_i, source_i in enumerate(keep):
        remap[source_i] = target_i
    lab_clipped = np.where(labels < 0, SOURCE_K, labels)      # -1 -> sentinel row of remap
    lab_target = remap[lab_clipped]                            # [n_rows,2] TARGET-space

    win_rows = _window_row_indices(bins_path, window_size)
    h1w = np.stack([lab_target[rows, 0] for rows in win_rows])   # [n_windows, T]
    h2w = np.stack([lab_target[rows, 1] for rows in win_rows])
    return h1w, h2w


def dump_real_row(model, cap, row_id, n_windows, window_size=512):
    row_dir = SIMVAL_EVAL / row_id
    suffix = "" if window_size == 512 else f"_w{window_size}"
    npy_path = row_dir / f"windowed_k24_fixdrop23{suffix}.npy"
    gametes_path = row_dir / "raw.npy.gametes.tsv"

    dataset_class, individual, coverage = row_id.split("__")
    kind = {"INBRED": "inbred", "HYB": "hybrid", "RIL": "ril"}[dataset_class.split("-")[1]]
    parents = split_individual_name(individual, kind)

    gametes = pd.read_csv(gametes_path, sep="\t").sort_values("gameteIndex")
    source_names = gametes["sampleName"].tolist()
    dropped_txt = row_dir / f"windowed_k24_fixdrop23{suffix}.dropped_idx.txt"
    dropped_idx = int(dropped_txt.read_text()) if dropped_txt.exists() else 23
    kept_names = [n for i, n in enumerate(source_names) if i != dropped_idx]
    name_to_col = {n: i for i, n in enumerate(kept_names)}

    full = np.load(npy_path, mmap_mode="r")
    feats_all = np.asarray(full[:, :, :K]).astype(np.float32)
    ext_vec = _founder_affinity(feats_all)  # [K,2], genome-wide, label-free

    n_total = full.shape[0]
    sel = _pick_window_idx(n_total, n_windows)
    X = torch.tensor(np.asarray(full[sel, :, :K]).astype(np.float32))
    B, T, _ = X.shape

    homo_scale_val = KIND_HOMO_SCALE[kind]
    homo_scale = torch.full((B,), homo_scale_val)
    ext_t = torch.tensor(ext_vec, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

    with torch.no_grad():
        emis_p, g, c = model(X, homo_scale, ext_t)
    H, emis_f, ext_bias_out = cap.H, cap.emis_f, cap.ext_bias_out

    # True founder pair: inbred/hybrid have ZERO breakpoints by this corpus's
    # own construction (truth.py builds them from the parent gVCF(s) directly,
    # no mosaic) -- see simval_truth_labels.py's bin_truth_labels docstring,
    # which documents the same fact. RIL genuinely mosaics -- real per-site
    # truth via bin_truth_labels/build_ril_mosaics (_window_ril_truth above).
    if kind == "ril":
        dataset_id = row_id.split("__")[0]
        h1w_all, h2w_all = _window_ril_truth(
            row_dir, source_names, dropped_idx, dataset_id, individual, parents,
            window_size=window_size)
        assert h1w_all.shape[0] == n_total, (
            f"{row_id}: truth windowing produced {h1w_all.shape[0]} windows, "
            f"features have {n_total} -- windowing mismatch")
        h1_true = torch.tensor(h1w_all[sel], dtype=torch.long)   # [B,T]
        h2_true = torch.tensor(h2w_all[sel], dtype=torch.long)
        pt = model.pair_table.numpy()
        true_pair_idx = torch.tensor(pt[h1_true.numpy(), h2_true.numpy()], dtype=torch.long)
    else:
        p_idx = [name_to_col[p] for p in parents if p in name_to_col]
        if len(p_idx) != len(parents):
            raise ValueError(f"{row_id}: a true parent was dropped from the panel ({parents})")
        h1_true = p_idx[0]
        h2_true = p_idx[-1]
        true_pair_idx = int(model.pair_table[h1_true, h2_true].item())

    result = _common_internals(
        model, X, H, emis_f, emis_p, g, c, ext_bias_out,
        homo_scale, true_pair_idx, h1_true, h2_true,
        het_scale_diagnostic=float(_het_scale(feats_all)), het_raw=cap.het)
    result.update(dict(
        row_id=row_id, source="real", kind=kind, parents=parents,
        homo_scale_used=homo_scale_val, n_windows_total=int(n_total),
        window_idx=sel.tolist(), ext_emb=ext_vec, founder_names=kept_names,
    ))
    return result


def dump_sim_group(model, cap, k_target, n_windows, seed=0):
    """One sim training individual with founder-count k_target (first match by
    subsampled scan), run through the model exactly as training did: fixed
    homo_scale is never adaptive for this checkpoint (no --adaptive-homo), so
    homo_scale=1.0 (full penalty, matching what the model actually saw during
    training) is used here, NOT the real-inference-time override."""
    a = np.load(SIM_DATA, mmap_mode="r")
    ind = np.load(SIM_IND)
    uniq_ind = np.unique(ind)
    G = int((ind == uniq_ind[0]).sum())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq_ind))
    chosen = None
    for i in order:
        s, e = i * G, (i + 1) * G
        h1 = np.asarray(a[s:e, :, K]).astype(np.int64)
        h2 = np.asarray(a[s:e, :, K + 1]).astype(np.int64)
        carried = np.unique(np.concatenate([h1.ravel(), h2.ravel()]))
        carried = carried[(carried >= 0) & (carried < K)]
        if len(carried) == k_target:
            chosen = i
            break
    if chosen is None:
        raise ValueError(f"no sim individual with k={k_target} found in scan")

    s, e = chosen * G, (chosen + 1) * G
    blk = np.asarray(a[s:e])
    feats_all = blk[:, :, :K].astype(np.float32)
    ext_vec = _founder_affinity(feats_all)

    sel = _pick_window_idx(G, n_windows)
    X = torch.tensor(feats_all[sel])
    h1w = blk[sel, :, K].astype(np.int64)
    h2w = blk[sel, :, K + 1].astype(np.int64)
    B, T, _ = X.shape

    homo_scale = torch.ones(B)  # sim training never had an adaptive scale
    ext_t = torch.tensor(ext_vec, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

    with torch.no_grad():
        emis_p, g, c = model(X, homo_scale, ext_t)
    H, emis_f, ext_bias_out = cap.H, cap.emis_f, cap.ext_bias_out

    h1_t = torch.tensor(np.clip(h1w, 0, K), dtype=torch.long)
    h2_t = torch.tensor(np.clip(h2w, 0, K), dtype=torch.long)
    true_pair = model.pair_table[h1_t, h2_t]  # [B,T] -- real per-site truth, sim only

    result = _common_internals(
        model, X, H, emis_f, emis_p, g, c, ext_bias_out,
        homo_scale, true_pair, h1_t, h2_t,
        het_scale_diagnostic=float(_het_scale(feats_all)), het_raw=cap.het)
    result.update(dict(
        row_id=f"sim_k{k_target}_ind{chosen:04d}", source="sim", kind=f"sim_k{k_target}",
        parents=None, homo_scale_used=1.0, n_windows_total=int(G),
        window_idx=sel.tolist(), ext_emb=ext_vec,
        founder_names=[f"F{i}" for i in range(K)],
    ))
    return result


def _common_internals(model, X, H, emis_f, emis_p, g, c, ext_bias_out,
                       homo_scale, true_pair, h1_true, h2_true, het_scale_diagnostic,
                       het_raw=None):
    """Shared post-processing: posteriors, homo-penalty A/B, marginal-vs-joint
    accuracy (H4), switch counts (H2), homo-mass with/without penalty (H3).
    `true_pair` may be a scalar pair index (real: constant genome-wide) or a
    [B,T] tensor (sim/RIL: genuinely varies per site). `h1_true`/`h2_true`
    match. `het_raw` is the captured per-locus het logit [B,T] -- only
    present (non-None) for a `--learned-het` checkpoint."""
    B, T, _ = X.shape
    pi, pj, nsw, stay_bonus = model.pi, model.pj, model.nsw_pair, model.stay_bonus
    homo_mask = model.homo_mask

    # Recompute pair emissions from the captured PER-FOUNDER emis_f ourselves,
    # with and without the homo penalty -- exact copy of forward()'s pairing
    # math (train_diploid.py:426-441), so both variants are self-consistent.
    # Two DIFFERENT penalty mechanisms depending on the checkpoint: a
    # `--learned-het` checkpoint uses a per-locus softplus(het) penalty (no
    # homo_scale/homo_penalty involved at all); the older fixed checkpoints
    # use homo_penalty*homo_scale. Branch exactly like forward() does, or the
    # self-check below will not match a learned-het checkpoint's real output.
    emis_p_raw = emis_f[..., pi] + emis_f[..., pj]           # no penalty
    if model.learned_het:
        assert het_raw is not None, "learned_het=True checkpoint but no het logit captured"
        het_pen = F.softplus(het_raw).unsqueeze(-1)           # [B,T,1] >= 0
        emis_p_pen = emis_p_raw - het_pen * homo_mask
    elif model.homo_penalty != 0.0:
        pen = model.homo_penalty * homo_scale.view(B, 1, 1)
        emis_p_pen = emis_p_raw - pen * homo_mask              # == emis_p from model()
    else:
        emis_p_pen = emis_p_raw

    # Self-check: our reconstruction must equal the model's own returned emis_p.
    max_diff = float((emis_p_pen - emis_p).abs().max())

    post_pen = compute_posteriors(emis_p_pen, c, nsw, stay_bonus)
    post_raw = compute_posteriors(emis_p_raw, c, nsw, stay_bonus)

    viterbi_pred = _dcrf_viterbi(emis_p, c, nsw, stay_bonus)   # [B,T]
    marginal_pred = post_pen.argmax(-1)                        # [B,T]

    # H2 -- path structure: decoded switch events per window (truth is 0 for
    # every real inbred/hybrid row by corpus construction; may be nonzero
    # for sim, which genuinely recombines).
    nsw_np = nsw.numpy()
    decoded_switches = (nsw_np[viterbi_pred[:, :-1].numpy(), viterbi_pred[:, 1:].numpy()] > 0).sum(-1)
    if torch.is_tensor(true_pair) and true_pair.dim() == 2:
        true_switches = (true_pair[:, :-1] != true_pair[:, 1:]).sum(-1).numpy()
    else:
        true_switches = np.zeros(B, dtype=int)

    # H3 -- het-prior effect: homozygous posterior mass with vs without the
    # fixed penalty (any homozygous state, including the null founder).
    homo_mass_pen = (post_pen * homo_mask[None, None, :]).sum(-1)   # [B,T]
    homo_mass_raw = (post_raw * homo_mask[None, None, :]).sum(-1)
    het_frac_decoded_pen = float((homo_mask[viterbi_pred] < 0.5).float().mean())
    viterbi_pred_nopen = _dcrf_viterbi(emis_p_raw, c, nsw, stay_bonus)
    het_frac_decoded_nopen = float((homo_mask[viterbi_pred_nopen] < 0.5).float().mean())

    # H4 -- joint-pair vs marginal-founder accuracy.
    if torch.is_tensor(true_pair) and true_pair.dim() == 2:
        true_pair_bt = true_pair
        t_lo, t_hi = torch.minimum(h1_true, h2_true), torch.maximum(h1_true, h2_true)
    else:
        true_pair_bt = torch.full((B, T), int(true_pair), dtype=torch.long)
        t_lo = torch.full((B, T), min(h1_true, h2_true), dtype=torch.long)
        t_hi = torch.full((B, T), max(h1_true, h2_true), dtype=torch.long)
    pair_acc_site = (viterbi_pred == true_pair_bt).float()          # [B,T]
    pred_lo, pred_hi = model.pi[viterbi_pred], model.pj[viterbi_pred]
    hap_acc_site = ((pred_lo == t_lo).float() + (pred_hi == t_hi).float()) / 2

    onehot_i = torch.zeros(model.P, K + 1)
    onehot_i[torch.arange(model.P), pi] = 1.0
    onehot_j = torch.zeros(model.P, K + 1)
    onehot_j[torch.arange(model.P), pj] = 1.0
    founder_marginal = post_pen @ (onehot_i + onehot_j)              # [B,T,K+1] expected chrom count
    top2 = founder_marginal[..., :K].topk(2, dim=-1).indices          # [B,T,2], real founders only
    m_lo = torch.minimum(top2[..., 0], top2[..., 1])
    m_hi = torch.maximum(top2[..., 0], top2[..., 1])
    marginal_hap_acc_site = ((m_lo == t_lo).float() + (m_hi == t_hi).float()) / 2

    post_entropy = -(post_pen.clamp_min(1e-9).log() * post_pen).sum(-1)  # [B,T]
    max_prob = post_pen.max(-1).values

    # PCA on this row's own H (top-8), for the encoder-drift figure.
    H_flat = H.reshape(-1, H.shape[-1])
    pca_coords, pca_var = pca_top8(H_flat)

    return dict(
        max_diff_selfcheck=max_diff,
        X=X.numpy(),                                            # [B,T,24] raw feature window (for fig06)
        H_norm=torch.linalg.norm(H, dim=-1).numpy(),          # [B,T]
        H_pca=pca_coords.reshape(B, T, -1),                    # [B,T,8]
        H_pca_explained_var=pca_var,
        gate=g.numpy(), recomb_cost=c.numpy(),
        emis_f=emis_f[..., :K].numpy(),                        # [B,T,24], drop null col
        ext_bias=(ext_bias_out.squeeze(-1)[:, :K].numpy() if ext_bias_out is not None else None),
        founder_marginal=founder_marginal[..., :K].numpy(),    # [B,T,24] expected chrom count per founder
        posterior_max_prob=max_prob.numpy(), posterior_entropy=post_entropy.numpy(),
        homo_mass_pen=homo_mass_pen.numpy(), homo_mass_raw=homo_mass_raw.numpy(),
        het_frac_decoded_pen=het_frac_decoded_pen, het_frac_decoded_nopen=het_frac_decoded_nopen,
        het_scale_diagnostic=het_scale_diagnostic,
        decoded_switches=decoded_switches, true_switches=true_switches,
        pair_acc=float(pair_acc_site.mean()), hap_acc=float(hap_acc_site.mean()),
        marginal_hap_acc=float(marginal_hap_acc_site.mean()),
        pair_acc_site=pair_acc_site.numpy(), hap_acc_site=hap_acc_site.numpy(),
        marginal_hap_acc_site=marginal_hap_acc_site.numpy(),
        viterbi_pred=viterbi_pred.numpy(), true_pair=(
            true_pair.numpy() if torch.is_tensor(true_pair) and true_pair.dim() == 2
            else np.full((B, T), int(true_pair))),
        pred_lo=pred_lo.numpy(), pred_hi=pred_hi.numpy(),
        true_lo=t_lo.numpy(), true_hi=t_hi.numpy(),
    )


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=DEFAULT_ROWS)
    ap.add_argument("--sim-k", nargs="+", type=int, default=[2, 13])
    ap.add_argument("--n-windows", type=int, default=24)
    ap.add_argument("--window-size", type=int, default=512,
                     help="Window size (T) of the real-data windowed .npy files "
                          "to dump. 512 uses windowed_k24_fixdrop23.npy; any other "
                          "value uses windowed_k24_fixdrop23_w<window_size>.npy.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                     help="Checkpoint to dump. Output goes to "
                          "results/model_internals/<ckpt-parent-dirname>/.")
    ap.add_argument("--out-tag", default=None,
                     help="Override the output subdirectory name (default: "
                          "the checkpoint's parent directory name, e.g. "
                          "'diploid-affinity-sim512-h3').")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_tag = args.out_tag or ckpt_path.parent.name
    out_dir = OUT_ROOT / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cap = load_model(ckpt_path)

    manifest = {
        "checkpoint": str(ckpt_path),
        "checkpoint_hparams": dict(
            num_parents=model.num_parents, d_model=model.encoder.d_model,
            founder_affinity=model.founder_affinity, homo_penalty=model.homo_penalty,
            learned_het=model.learned_het),
        "n_windows_per_row": args.n_windows,
        "K_real_founders": K,
        "P_pairs": model.P,
        "note_learned_het": (
            "learned_het=True on this checkpoint: heterozygosity enters as a "
            "PER-LOCUS softplus(het_head(H)) penalty, ignoring homo_scale/"
            "homo_penalty entirely." if model.learned_het else
            "learned_het=False on this checkpoint: there is no per-locus het head. "
            "Heterozygosity enters ONLY as homo_penalty * homo_scale * homo_mask, "
            "subtracted from every homozygous pair-state's emission."),
        "dim_meanings": {
            "H_norm": "[n_windows,T] L2 norm of encoder hidden state per site",
            "H_pca": "[n_windows,T,8] top-8 PCA of this row's H, row-local basis",
            "gate": "[n_windows,T] sigmoid gate g in [0,1]",
            "recomb_cost": "[n_windows,T] softplus recomb-cost c (switch cost, NOT a probability)",
            "emis_f": "[n_windows,T,24] per-founder emission logit (post-gate, post-ext_bias)",
            "ext_bias": "[n_windows,24] learned per-founder affinity logit offset (constant over T)",
            "posterior_max_prob": "[n_windows,T] max_p P(pair=p | reads), forward-backward",
            "posterior_entropy": "[n_windows,T] entropy of the pair posterior",
            "homo_mass_pen": "[n_windows,T] posterior mass on any homozygous pair, WITH the fixed homo penalty applied",
            "homo_mass_raw": "[n_windows,T] same, penalty removed (H3 discriminator)",
            "decoded_switches": "[n_windows] # sites where Viterbi path switches >=1 chromosome",
            "true_switches": "[n_windows] same for ground truth (H2 discriminator; 0 for real inbred/hybrid by construction)",
            "pair_acc / hap_acc": "scalar, joint Viterbi-pair vs per-haplotype accuracy",
            "marginal_hap_acc": "scalar, top-2-by-posterior-marginal founder accuracy (H4 discriminator)",
        },
        "rows": [],
    }

    for row_id in args.rows:
        print(f"Dumping real row {row_id}...")
        res = dump_real_row(model, cap, row_id, args.n_windows, window_size=args.window_size)
        _save(res, out_dir)
        manifest["rows"].append(_row_summary(res))

    for k_target in args.sim_k:
        print(f"Dumping sim comparison group k={k_target}...")
        res = dump_sim_group(model, cap, k_target, args.n_windows, args.seed)
        _save(res, out_dir)
        manifest["rows"].append(_row_summary(res))

    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {out_dir / 'manifest.json'}")
    print(json.dumps(manifest["rows"], indent=2))


def _row_summary(res):
    return dict(
        row_id=res["row_id"], source=res["source"], kind=res["kind"], parents=res["parents"],
        homo_scale_used=res["homo_scale_used"], n_windows_total=res["n_windows_total"],
        max_diff_selfcheck=res["max_diff_selfcheck"],
        pair_acc=res["pair_acc"], hap_acc=res["hap_acc"], marginal_hap_acc=res["marginal_hap_acc"],
        mean_decoded_switches=float(np.mean(res["decoded_switches"])),
        mean_true_switches=float(np.mean(res["true_switches"])),
        het_frac_decoded_pen=res["het_frac_decoded_pen"],
        het_frac_decoded_nopen=res["het_frac_decoded_nopen"],
        het_scale_diagnostic=res["het_scale_diagnostic"],
        founder_names=res["founder_names"],
    )


def _save(res, out_dir):
    out_path = out_dir / f"{res['row_id']}.npz"
    arrs = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(out_path, **arrs)
    print(f"  wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB)  "
          f"self-check max_diff={res['max_diff_selfcheck']:.2e}  "
          f"pair_acc={res['pair_acc']:.4f} hap_acc={res['hap_acc']:.4f} "
          f"marginal_hap_acc={res['marginal_hap_acc']:.4f}")


if __name__ == "__main__":
    main()
