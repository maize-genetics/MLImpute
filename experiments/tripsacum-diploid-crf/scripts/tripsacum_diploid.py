#!/usr/bin/env python
"""
Synthetic-diploid baseline for the Tripsacum pangenome: simulate reads (wgsim) directly
from two assembly FASTAs, combine them into one "individual" with no recombination (truth
= (assemblyA, assemblyB) at every site by construction), run the combined reads through
`ropebwt3 refmap` -> windowed-matrix -> a diploid GRITS-CRF, and score. Supports both the
plain diploid-sim512-h3 checkpoint (`crf/eval.py::evaluate_diploid`) and the
diploid-affinity-sim512-h3 checkpoint (`evaluate_diploid_with_affinity`, this file --
computes one genome-wide `_founder_affinity` vector from this run's own data and feeds it
as `ext_emb`, the diploid equivalent of the "single real individual, no
--windows-per-individual grouping" pattern `crf/eval.py`'s own docstring already documents
for the haploid case but never implements for diploid mode).

This generalizes grits_workdir/scripts/nam_diploid.py two ways:
  1. Reads are SIMULATED directly from assembly FASTA (wgsim), not subsampled from
     pre-existing real WGS files -- there is no ground truth for the real Tripsacum WGS
     reads, so a controlled simulated test is needed first.
  2. Runs on the Tripsacum 18-sample pangenome (tripsacumChrIndex.{fmd,lift}, built this
     session) instead of NAM's 25-founder maize pangenome.

Checkpoint dimensionality mismatch (verified by inspecting the checkpoint's saved
hyperparameters/state_dict directly): checkpoints/diploid-sim512-h3/last.ckpt is hard-fixed
to num_parents=24 (its pair_table/pi/pj/nsw_pair/homo_mask buffers are deterministic
functions of K=24, built once at construction -- train_diploid.py's build_pair_tables(K)).
The Tripsacum panel has only 18 samples. ropebwt_npy_to_matrix.py's --target-num-parents
only trims DOWN (raises ValueError if target > source K), so it cannot bridge 18->24. This
script pads the windowed K=18 matrix up to K=24 with 6 permanently-zero founder columns
(pad_to_24()) before evaluation. This is architecturally sound because the model has no
per-founder-identity learned weights anywhere (the per-site encoder cell is a single shared
Linear(1,256) applied to every founder column, and the pair-space buffers are pure index
bookkeeping) -- it was trained on simulate_alleles.py's fully abstract, index-only founder
regime, so a handful of permanently-empty founder columns are within its training
distribution (structurally identical to a real founder that's simply absent genome-wide).

The padded/masked feature arrays don't depend on which checkpoint scores them, so both
model variants reuse the exact same cached reads/refmap/windowing/padding per pair -- only
the final scoring step differs. Plain-model results/files are untouched by the affinity
modes (separate result files: tripsacum_diploid_affinity.{tsv,md}).

Pairs were chosen by measured relatedness (mash, tripsacum/relatedness/all_pairs_dist.tsv):
starter C009-T009 x C011-T007 (dist=0.009366), then the closest-distance matches among all
other unordered pairs of the 14 non-reference assemblies.

Usage:
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_diploid.py list
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_diploid.py one <A> <B> <depth>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_diploid.py all
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_diploid.py report
    # affinity-model variants (same args, checkpoints/diploid-affinity-sim512-h3):
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_diploid.py one-affinity <A> <B> <depth>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_diploid.py all-affinity
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/tripsacum_diploid.py report-affinity
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PY = sys.executable
THREADS = "50"

# ---- ropebwt3 / Tripsacum pangenome index (built earlier this session) -------------
BIN = Path("/local/workdir/zrm22/HackathonJun2026/ropebwt_refMap/ropebwt3-phg/"
           ".claude/worktrees/refmap-ps4g-numpy/ropebwt3")
IDX_DIR = Path("/workdir/zrm22/HackathonJun2026/grits_workdir/tripsacum/ropebwt_index")
FMD = IDX_DIR / "tripsacumChrIndex.fmd"
LIFT = IDX_DIR / "tripsacumChrIndex.lift"
REF_PREFIX = "Td-FL-9056069-6-REFERENCE-PanAnd-2.0a_"

ASM_DIR = Path("/workdir/zrm22/HackathonJun2026/grits_workdir/tripsacum/assemblies")
WGSIM = Path("/programs/samtools-1.20/bin/wgsim")

# ---- CRF / checkpoint --------------------------------------------------------------
CRF_SRC = Path("/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
WINDOW_SCRIPT = CRF_SRC / "python/crf/ropebwt_npy_to_matrix.py"
DIPLOID_CKPT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/"
                     "checkpoints/diploid-sim512-h3/last.ckpt")
AFFINITY_CKPT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/checkpoints/"
                      "diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt")

sys.path.insert(0, str(CRF_SRC))

SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/tripsacum_diploid")
RESULTS_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/tripsacum_diploid.tsv")
RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/tripsacum_diploid.md")
AFFINITY_RESULTS_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/"
                             "tripsacum_diploid_affinity.tsv")
AFFINITY_RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/"
                            "tripsacum_diploid_affinity.md")

N_TARGET_PARENTS = 18   # actual Tripsacum pangenome panel size
N_MODEL_PARENTS = 24    # checkpoint's fixed founder-column count (padded up from 18)

# Td-FL-9056069-6-REFERENCE-PanAnd-2.0a chromosome lengths, extracted from
# tripsacumChrIndex.fmd.len.gz this session (18 canonical chr1..chr18 entries).
CHR_LENGTHS = [
    ("chr1", 274830655), ("chr2", 229128910), ("chr3", 179943459), ("chr4", 176900500),
    ("chr5", 156931990), ("chr6", 147617370), ("chr7", 141981561), ("chr8", 141375430),
    ("chr9", 121026982), ("chr10", 120589626), ("chr11", 119166894), ("chr12", 110505784),
    ("chr13", 107689641), ("chr14", 103922234), ("chr15", 99826684), ("chr16", 97159109),
    ("chr17", 83879775), ("chr18", 74345949),
]

# Chosen by mash distance (tripsacum/relatedness/all_pairs_dist.tsv): starter pair plus
# the 3 closest-distance matches among all other unordered pairs of the 14 assemblies
# (dist=0.009366 target; matches at 0.009406, 0.009206, and the closest fully-independent-
# member pair at 0.006910).
PAIRS = [
    ("C009-T009", "C011-T007"),  # starter, dist=0.009366
    ("C009-T009", "C027-T007"),  # dist=0.009406 (|diff|=0.00004)
    ("C009-T009", "C050-T007"),  # dist=0.009206 (|diff|=0.00016)
    ("C076-T198", "C081-T199"),  # dist=0.006910 (|diff|=0.00246) -- independent members
]

DEPTHS = [250_000]  # reads/haplotype; lowest depth in the prior NAM sweep scored best


def assembly_path(sample):
    p = ASM_DIR / f"{sample}.fa"
    if not p.exists():
        raise FileNotFoundError(f"no assembly for {sample!r}: {p}")
    return p


def combo_name(founders, depth):
    """founders: list of sample names (2 for the pair tests, N for the nway recombination
    tests). "x".join(...) reproduces today's "AxB" naming exactly for a 2-element list, so
    existing pair-run directories stay addressable under the same names."""
    return f"{'x'.join(founders)}_{depth // 1000}k"


def simulate_reads(sample, depth, outdir):
    """wgsim-simulate `depth` read pairs directly from the assembly FASTA; keep R1 only
    (matches the WGSReads/*_1.clean.fq.gz R1-only convention -- refmap has no paired-end
    logic, just independent whole-read placement). -r0/-R0 = no germline mutations/indels
    injected (reads carry only sequencing noise, not fabricated variants, preserving the
    known-by-construction truth). Deterministic seed for cache/resumability."""
    outdir.mkdir(parents=True, exist_ok=True)
    r1 = outdir / f"{sample}_{depth}.R1.fastq"
    if r1.exists():
        return r1
    r1_tmp = outdir / f"{sample}_{depth}.R1.fastq.tmp"
    r2_tmp = outdir / f"{sample}_{depth}.R2.fastq.tmp"
    seed = abs(hash(sample)) % (2 ** 31)
    cmd = [str(WGSIM), "-e", "0.001", "-r", "0", "-R", "0",
           "-1", "100", "-2", "100", "-N", str(depth), "-S", str(seed),
           str(assembly_path(sample)), str(r1_tmp), str(r2_tmp)]
    print(f"  [{sample}] simulating {depth:,} read pairs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    r1_tmp.rename(r1)
    r2_tmp.unlink(missing_ok=True)
    return r1


def combine_reads(founders, depth, outdir):
    """Simulate each founder's reads and concatenate -> one FASTQ ("individual") with every
    founder's reads interleaved at the file level (no recombination -- truth is exactly the
    constant N-way founder set everywhere; the recombination scripts mask this down to a
    per-site-varying pair/N-way ancestry afterward). `founders`: list of 2 (pair tests) or
    more (nway tests) sample names."""
    outdir.mkdir(parents=True, exist_ok=True)
    combined = outdir / "combined.fastq"
    if combined.exists():
        return combined
    r1_paths = [simulate_reads(f, depth, outdir) for f in founders]
    with open(combined, "wb") as out_f:
        for r1 in r1_paths:
            with open(r1, "rb") as in_f:
                out_f.write(in_f.read())
    return combined


def make_labels_bed(outdir):
    """18-row BED over the reference chromosomes; col4 is a placeholder -- the label
    columns get overwritten with the known-by-construction (idxA, idxB) pair after
    windowing, so its value here is irrelevant, only its presence/format matters."""
    out_path = outdir / "labels.bed"
    if out_path.exists():
        return out_path
    with open(out_path, "w") as f:
        for chrom, length in CHR_LENGTHS:
            f.write(f"{REF_PREFIX}{chrom}\t0\t{length}\tPLACEHOLDER\n")
    return out_path


def run_refmap_combined(combined_fastq, outdir):
    npy_path = outdir / "raw.npy"
    ps4g_path = outdir / "raw.ps4g"
    tsv_path = outdir / "raw.tsv"
    log_path = outdir / "raw.log"
    labels_path = make_labels_bed(outdir)

    if npy_path.exists() and tsv_path.exists():
        print(f"  [{outdir.name}] refmap output already exists, skipping run")
        return npy_path

    cmd = [str(BIN), "refmap", f"--ref-prefix={REF_PREFIX}", "--max-occ=-1",
           f"--lift={LIFT}", "-t", THREADS,
           f"--label-bed={labels_path}", f"--ps4g={ps4g_path}", f"--npy={npy_path}",
           str(FMD), str(combined_fastq)]
    print(f"  [{outdir.name}] running: {' '.join(cmd)}")
    t0 = time.time()
    with open(tsv_path, "w") as out_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
    log_path.write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"refmap failed for {outdir.name}: {proc.stderr[-2000:]}")
    print(f"  [{outdir.name}] refmap done in {time.time() - t0:.1f}s")
    return npy_path


def window(raw_npy, outdir):
    """crf/ropebwt_npy_to_matrix.py at the true panel size (18) -- no --target-num-parents,
    it can only trim DOWN, and we need to go the other way (pad up, done separately in
    pad_to_24)."""
    bins_path = outdir / "raw.npy.bins.tsv"
    gametes_path = outdir / "raw.npy.gametes.tsv"
    out_path = outdir / "windowed_k18.npy"

    if out_path.exists():
        return out_path, gametes_path

    cmd = [PY, str(WINDOW_SCRIPT), f"--npy={raw_npy}", f"--bins={bins_path}",
           f"--gametes={gametes_path}", f"--num-parents={N_TARGET_PARENTS}",
           "--window-size=512", f"--out={out_path}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"windowing failed for {outdir.name}: {proc.stderr[-2000:]}")
    print(proc.stdout)
    return out_path, gametes_path


def discover_panel_order(gametes_tsv):
    """Read the gameteIndex/sampleName mapping ropebwt_npy_to_matrix.py's --gametes sidecar
    emits, rather than hardcoding it -- safer with 18 long hyphenated sample names than
    assuming a particular order (e.g. alphabetical) matches the pangenome's internal
    founder-column order."""
    df = pd.read_csv(gametes_tsv, sep="\t")
    assert list(df.columns) == ["gameteIndex", "sampleName"], \
        f"unexpected gametes.tsv columns: {list(df.columns)}"
    assert len(df) == N_TARGET_PARENTS, \
        f"expected {N_TARGET_PARENTS} founders in gametes.tsv, got {len(df)}"
    return {row.sampleName: int(row.gameteIndex) for row in df.itertuples()}


def pad_to_24(windowed_k18_path, outdir):
    """Append 6 permanently-zero founder feature columns to bridge the true 18-founder
    panel to the checkpoint's fixed 24-founder input. Pre-overwrite label values are not
    preserved carefully here (no remap of the K=18 windowing script's own "unknown" sentinel,
    value 18, into K=24 space) because write_diploid_labels immediately overwrites both
    label columns with the known-by-construction truth afterward -- the original label
    values are never read."""
    out_path = outdir / "padded_k24.npy"
    if out_path.exists():
        return out_path
    arr = np.load(windowed_k18_path)
    N, T, C = arr.shape
    K18 = C - 2
    assert K18 == N_TARGET_PARENTS, f"expected {N_TARGET_PARENTS} founder cols, got {K18}"
    padded = np.zeros((N, T, N_MODEL_PARENTS + 2), dtype=arr.dtype)
    padded[:, :, :K18] = arr[:, :, :K18]
    padded[:, :, N_MODEL_PARENTS] = arr[:, :, K18]
    padded[:, :, N_MODEL_PARENTS + 1] = arr[:, :, K18 + 1]
    np.save(out_path, padded)
    return out_path


def write_diploid_labels(padded_npy_path, idxA, idxB, out_path):
    """Overwrite the two label columns with the known-by-construction (idxA, idxB) pair
    at every site -- same pattern as nam_diploid.py's write_diploid_labels."""
    if out_path.exists():
        return out_path
    arr = np.load(padded_npy_path)
    K = arr.shape[-1] - 2
    assert K == N_MODEL_PARENTS
    arr = arr.copy()
    arr[:, :, K] = idxA
    arr[:, :, K + 1] = idxB
    np.save(out_path, arr)
    return out_path


def founder_density_stats(k18_npy_path, founder_idx):
    a = np.asarray(np.load(k18_npy_path, mmap_mode="r"))
    K = a.shape[-1] - 2
    feat = a[:, :, :K].reshape(-1, K)
    nz_row = feat != 0
    self_cov = nz_row[:, founder_idx].mean() * 100
    return self_cov


def evaluate_diploid_with_affinity(model, ds, device, batch_size, num_workers, ext_vec):
    """Same metric computation as crf/eval.py::evaluate_diploid, but broadcasts one
    genome-wide affinity vector (this run's own data, via _founder_affinity) to every
    window -- crf/eval.py already documents and implements this exact "single real
    individual, no --windows-per-individual grouping" pattern for the HAPLOID case
    (its module docstring), but never wires ext_emb through for diploid mode at all
    (confirmed: the diploid branch always calls model(X) with no ext_emb). This is the
    diploid equivalent, kept local rather than editing the shared crf/ library."""
    import torch
    from torch.utils.data import DataLoader
    from python.crf.train_diploid import _dcrf_viterbi

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    pair_correct = hap_correct = total = 0
    homo_pred = 0
    ext_t = torch.tensor(ext_vec, dtype=torch.float32, device=device)  # [K,2]
    for batch in loader:
        X = batch["input_embeds"].to(device)
        h1 = batch["h1"].to(device)
        h2 = batch["h2"].to(device)
        ext_batch = ext_t.unsqueeze(0).expand(X.shape[0], -1, -1)      # [B,K,2]
        emis_p, _, c = model(X, None, ext_batch)
        pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
        pair_true = model.pair_table[h1, h2]
        pair_correct += (pred == pair_true).sum().item()
        pred_lo, pred_hi = model.pi[pred], model.pj[pred]
        t_lo = torch.minimum(h1, h2)
        t_hi = torch.maximum(h1, h2)
        hap_correct += ((pred_lo == t_lo).sum() + (pred_hi == t_hi).sum()).item()
        homo_pred += (pred_lo == pred_hi).sum().item()
        total += pair_true.numel()
    return {"pair_acc": pair_correct / total, "hap_acc": hap_correct / (2 * total),
            "homo_pred": homo_pred / total, "n": total}


def run_diploid_eval(data_path, tag, device, ckpt_path=DIPLOID_CKPT):
    """In-process reuse of crf/train_diploid.py + crf/eval.py, same pattern as
    nam_diploid.py's run_diploid_eval. Auto-detects an affinity-trained checkpoint
    (mirrors crf/eval.py's own `needs_affinity = getattr(model, "ext_dim", 0) > 0` idiom
    for haploid, adapted to the attribute GRITSCRFDiploid actually stores) and computes/
    feeds its ext_emb accordingly; behaves exactly as before for the plain model."""
    import torch
    from python.crf.train_diploid import GRITSCRFDiploid, make_diploid_splits, _founder_affinity
    from python.crf.eval import evaluate_diploid

    model = GRITSCRFDiploid.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)
    _, _, test_ds = make_diploid_splits(str(data_path), num_parents=N_MODEL_PARENTS,
                                         val_frac=0.0, test_frac=1.0)
    if len(test_ds) == 0:
        return dict(n=0, pair_acc=float("nan"), hap_acc=float("nan"), homo_pred=float("nan"))

    needs_affinity = getattr(model, "founder_affinity", False)
    if needs_affinity:
        feats = np.asarray(np.load(data_path))[:, :, :N_MODEL_PARENTS]
        ext_vec = _founder_affinity(feats)  # [K,2], label-free -> safe at inference
        r = evaluate_diploid_with_affinity(model, test_ds, device, batch_size=128,
                                            num_workers=4, ext_vec=ext_vec)
    else:
        r = evaluate_diploid(model, test_ds, device, batch_size=128, num_workers=4)
    print(f"  [{tag}] affinity={needs_affinity}  pair_acc={r['pair_acc']:.4f}  "
          f"hap_acc={r['hap_acc']:.4f}  homo_pred={r['homo_pred']:.4f}  n={r['n']:,}")
    return r


RESULT_COLS = ["assemblyA", "assemblyB", "depth_per_hap", "n_placed", "n_unplaced",
               "self_cov_A_pct", "self_cov_B_pct", "het_frac", "n_sites",
               "pair_acc", "hap_acc", "homo_pred"]


def write_header_if_needed(results_tsv=RESULTS_TSV):
    results_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not results_tsv.exists():
        results_tsv.write_text("\t".join(RESULT_COLS) + "\n")


def already_recorded(a, b, depth, results_tsv=RESULTS_TSV):
    if not results_tsv.exists():
        return False
    key = f"{a}\t{b}\t{depth}\t"
    with open(results_tsv) as f:
        return any(line.startswith(key) for line in f)


def run_one(a, b, depth, device, force=False, ckpt_path=DIPLOID_CKPT, results_tsv=RESULTS_TSV):
    if not force and already_recorded(a, b, depth, results_tsv):
        print(f"[{a}x{b}@{depth}] already in results TSV, skipping entirely")
        return

    name = combo_name([a, b], depth)
    outdir = SCRATCH / name
    print(f"\n=== {name} ===")

    combined = combine_reads([a, b], depth, outdir)
    raw_npy = run_refmap_combined(combined, outdir)

    n_placed = n_unplaced = 0
    with open(outdir / "raw.tsv") as f:
        for line in f:
            status = line.split("\t", 3)[2]
            if status in ("PLACED", "EXACT"):
                n_placed += 1
            elif status == "UNPLACED":
                n_unplaced += 1

    if n_placed == 0:
        print(f"  [{name}] n_placed=0 -- degenerate, recording NaN metrics")
        row = dict(assemblyA=a, assemblyB=b, depth_per_hap=depth,
                    n_placed=n_placed, n_unplaced=n_unplaced,
                    self_cov_A_pct=float("nan"), self_cov_B_pct=float("nan"),
                    het_frac=float("nan"), n_sites=0, pair_acc=float("nan"),
                    hap_acc=float("nan"), homo_pred=float("nan"))
    else:
        k18_npy, gametes_tsv = window(raw_npy, outdir)
        panel = discover_panel_order(gametes_tsv)
        if a not in panel or b not in panel:
            raise RuntimeError(f"{name}: {a!r}/{b!r} not found in discovered panel "
                                f"{sorted(panel)}")
        idxA, idxB = panel[a], panel[b]

        self_cov_A = founder_density_stats(k18_npy, idxA)
        self_cov_B = founder_density_stats(k18_npy, idxB)

        padded = pad_to_24(k18_npy, outdir)
        diploid_npy = write_diploid_labels(padded, idxA, idxB, outdir / "diploid_k24.npy")

        arr = np.load(diploid_npy, mmap_mode="r")
        K = arr.shape[-1] - 2
        het_frac = float((arr[:, :, K] != arr[:, :, K + 1]).mean())
        if het_frac != 1.0:
            raise RuntimeError(f"{name}: expected 100% het by construction, got "
                                f"{het_frac * 100:.2f}% -- label overwrite bug")

        r = run_diploid_eval(diploid_npy, name, device, ckpt_path=ckpt_path)
        row = dict(assemblyA=a, assemblyB=b, depth_per_hap=depth,
                    n_placed=n_placed, n_unplaced=n_unplaced,
                    self_cov_A_pct=self_cov_A, self_cov_B_pct=self_cov_B,
                    het_frac=het_frac, n_sites=r["n"], pair_acc=r["pair_acc"],
                    hap_acc=r["hap_acc"], homo_pred=r["homo_pred"])

    write_header_if_needed(results_tsv)
    if force and already_recorded(a, b, depth, results_tsv):
        lines = results_tsv.read_text().splitlines(keepends=True)
        key = f"{a}\t{b}\t{depth}\t"
        keep = [l for l in lines if not l.startswith(key)]
        results_tsv.write_text("".join(keep))
    with open(results_tsv, "a") as f:
        f.write("\t".join(str(row[c]) for c in RESULT_COLS) + "\n")
    print(f"[{name}] n_placed={n_placed}  self_cov_A={row['self_cov_A_pct']}  "
          f"self_cov_B={row['self_cov_B_pct']}  pair_acc={row['pair_acc']}  "
          f"hap_acc={row['hap_acc']}  homo_pred={row['homo_pred']}")


def _markdown_table(df, float_cols):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4f}" if (c in float_cols and isinstance(v, float)) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_report(results_tsv=RESULTS_TSV, results_md=RESULTS_MD, affinity=False):
    df = pd.read_csv(results_tsv, sep="\t")
    df = df.sort_values(["assemblyA", "assemblyB", "depth_per_hap"]).reset_index(drop=True)
    float_cols = {"self_cov_A_pct", "self_cov_B_pct", "het_frac", "pair_acc",
                  "hap_acc", "homo_pred"}
    table_md = _markdown_table(df, float_cols)

    model_desc = ("the **affinity-conditioned** (`founder_affinity=True`, "
                   "`checkpoints/diploid-affinity-sim512-h3`) diploid GRITS-CRF, fed a "
                   "genome-wide `_founder_affinity` vector computed from this run's own "
                   "data (the same 'single real individual' pattern `crf/eval.py` already "
                   "documents for the haploid case, added here for diploid -- see "
                   "`run_diploid_eval`/`evaluate_diploid_with_affinity`)"
                   if affinity else
                   "the plain (non-affinity; `founder_affinity=False`) diploid GRITS-CRF "
                   "via `crf/eval.py::evaluate_diploid`")
    lines = [
        f"# Synthetic-diploid Tripsacum baseline (simulated reads, "
        f"{'affinity' if affinity else 'plain'} diploid CRF)\n",
        "Each row simulates reads directly from two assembly FASTAs (`wgsim -e 0.001 -r 0 "
        "-R 0 -1 100 -2 100`, R1 only kept), combines them into one FASTQ with no "
        "recombination (truth = `(assemblyA, assemblyB)` at every site by construction), "
        "runs the combined reads through `ropebwt3 refmap` against the Tripsacum 18-sample "
        "pangenome (`tripsacumChrIndex.{fmd,lift}`, `--ref-prefix="
        f"{REF_PREFIX}`), windows at the true panel size K=18 "
        "(`crf/ropebwt_npy_to_matrix.py --window-size=512`), then pads to K=24 with 6 "
        "permanently-zero founder columns to match the checkpoint's fixed 24-founder "
        f"architecture. Scored with {model_desc}.\n\n"
        "Pairs were chosen by measured genetic relatedness (`mash dist`, "
        "`tripsacum/relatedness/all_pairs_dist.tsv` -- no accession metadata exists, and no "
        "distance tool was previously installed): the starter pair C009-T009 x C011-T007 "
        "(dist=0.009366), then the closest-distance matches among all other unordered pairs "
        "of the 14 non-reference assemblies (0.009406, 0.009206, and the closest fully-"
        "independent-member pair at 0.006910).\n\n"
        f"{table_md}\n",
    ]
    results_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {results_md}")
    print(df.to_string(index=False))


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "", file=sys.stderr)
        sys.exit(1)
    mode = sys.argv[1]

    if mode == "list":
        needed = {a for a, b in PAIRS} | {b for a, b in PAIRS}
        for name in sorted(needed):
            try:
                assembly_path(name)
                status = "OK"
            except FileNotFoundError:
                status = "MISSING"
            print(f"{name:<12} {status}")
        print(f"\nPairs: {PAIRS}")
        print(f"Depths (per haplotype): {DEPTHS}")
        return

    if mode == "report":
        write_report()
        return

    if mode == "report-affinity":
        write_report(AFFINITY_RESULTS_TSV, AFFINITY_RESULTS_MD, affinity=True)
        return

    if mode in ("one", "one-affinity"):
        if len(sys.argv) < 5:
            raise SystemExit(f"usage: tripsacum_diploid.py {mode} <A> <B> <depth>")
        a, b, depth = sys.argv[2], sys.argv[3], int(sys.argv[4])
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = AFFINITY_CKPT if mode == "one-affinity" else DIPLOID_CKPT
        tsv = AFFINITY_RESULTS_TSV if mode == "one-affinity" else RESULTS_TSV
        write_header_if_needed(tsv)
        run_one(a, b, depth, device, force=True, ckpt_path=ckpt, results_tsv=tsv)
    elif mode in ("all", "all-affinity"):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = AFFINITY_CKPT if mode == "all-affinity" else DIPLOID_CKPT
        tsv = AFFINITY_RESULTS_TSV if mode == "all-affinity" else RESULTS_TSV
        write_header_if_needed(tsv)
        for a, b in PAIRS:
            for depth in DEPTHS:
                run_one(a, b, depth, device, ckpt_path=ckpt, results_tsv=tsv)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    print(f"\nDone. Results in {AFFINITY_RESULTS_TSV if 'affinity' in mode else RESULTS_TSV}")


if __name__ == "__main__":
    main()
