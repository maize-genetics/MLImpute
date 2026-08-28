#!/usr/bin/env python
"""
Synthetic-diploid baseline for the cassava (Manihot esculenta) pangenome: simulate reads
(wgsim) directly from two assembly FASTAs, combine them into one "individual" with no
recombination (truth = (assemblyA, assemblyB) at every site by construction), run the
combined reads through `ropebwt3 refmap` -> windowed-matrix -> a diploid GRITS-CRF, and
score. Supports both the plain diploid-sim512-h3 checkpoint (`crf/eval.py::evaluate_diploid`)
and the diploid-affinity-sim512-h3 checkpoint (`evaluate_diploid_with_affinity`, this file).

Direct adaptation of grits_workdir/scripts/tripsacum_diploid.py -- reuses its wgsim ->
refmap -> window -> score flow essentially unchanged. See that file's docstring for the
general rationale (why simulate reads at all, why the affinity ext_emb is computed the
way it is, etc.); this docstring only covers what's cassava-specific.

**The one substantive difference from Tripsacum**: Tripsacum's real panel (18 founders)
was smaller than the checkpoint's fixed num_parents=24, so tripsacum_diploid.py PADS UP
with 6 permanently-zero founder columns. Cassava's indexed panel is 80 founders (~40
diploid accessions x 2 haplotypes each) -- much LARGER than 24. `ropebwt_npy_to_matrix.py
--target-num-parents` can only trim down, and does so by dropping the least-covered
founders genome-wide (fewest rows with nonzero read support). Since simulated reads for
this run come from exactly two founders, those two founders are always the most-covered
in the whole panel -- so instead of the module-level `--target-num-parents` flag (which
doesn't know which founders our reads came from and doesn't guarantee our pair survives),
this script does its own selection (`select_to_24`) that FORCE-INCLUDES the true pair and
fills the remaining 22 slots with the next most-covered founders. This is the faithful
analog of Tripsacum, which effectively kept only the founders that ever received reads.

**Two kinds of pairs are tested** (tagged via the `kind` column in results), because
cassava's panel is hap-resolved (most accessions have both haplotypes assembled):
  - "within": hap1 x hap2 of the SAME accession -- a genuinely real heterozygous
    individual, not a synthetic cross. Cassava is the only one of our datasets (NAM,
    Tripsacum) where this is possible.
  - "cross":  Tripsacum-style pairs across DIFFERENT accessions, chosen by mash distance
    (cassava/relatedness/all_pairs_dist.tsv, built for this pipeline -- no relatedness
    data existed for cassava previously).

Usage:
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/cassava_diploid.py list
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/cassava_diploid.py one <A> <B> <depth>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/cassava_diploid.py all
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/cassava_diploid.py report
    # affinity-model variants (same args, checkpoints/diploid-affinity-sim512-h3):
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/cassava_diploid.py one-affinity <A> <B> <depth>
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/cassava_diploid.py all-affinity
    /home/zrm22/mambaforge/envs/phg-ml/bin/python grits_workdir/scripts/cassava_diploid.py report-affinity
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PY = sys.executable
THREADS = "50"

# ---- ropebwt3 / cassava pangenome index (already built) ----------------------------
BIN = Path("/local/workdir/zrm22/HackathonJun2026/ropebwt_refMap/ropebwt3-phg/"
           ".claude/worktrees/refmap-ps4g-numpy/ropebwt3")
IDX_DIR = Path("/workdir/zrm22/HackathonJun2026/grits_workdir/cassava/ropebwt_index")
FMD = IDX_DIR / "cassavaChrIndex.fmd"
LIFT = IDX_DIR / "cassavaChrIndex.lift"
REF_PREFIX = "Mesculenta-671-v8-0_"

ASM_DIR = Path("/workdir/zrm22/HackathonJun2026/grits_workdir/cassava/"
               "fasta_for_index/no_wgs_available")
WGSIM = Path("/programs/samtools-1.20/bin/wgsim")

# ---- CRF / checkpoint --------------------------------------------------------------
CRF_SRC = Path("/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
WINDOW_SCRIPT = CRF_SRC / "python/crf/ropebwt_npy_to_matrix.py"
DIPLOID_CKPT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/"
                     "checkpoints/diploid-sim512-h3/last.ckpt")
AFFINITY_CKPT = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/checkpoints/"
                      "diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt")

sys.path.insert(0, str(CRF_SRC))

SCRATCH = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/scratch/cassava_diploid")
RESULTS_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/cassava_diploid.tsv")
RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/cassava_diploid.md")
AFFINITY_RESULTS_TSV = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/"
                             "cassava_diploid_affinity.tsv")
AFFINITY_RESULTS_MD = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results/"
                            "cassava_diploid_affinity.md")

N_TARGET_PARENTS = 80   # full indexed cassava panel size (windowing keeps all 80)
N_MODEL_PARENTS = 24    # checkpoint's fixed founder-column count (selected down from 80)

# Mesculenta-671-v8-0 (Phytozome cassava v8) chromosome lengths, extracted from
# cassavaChrIndex.fmd.len.gz (18 canonical Chromosome01..18 entries; the index also
# contains thousands of small unplaced scaffolds, deliberately excluded here).
CHR_LENGTHS = [
    ("Chromosome01", 42998274), ("Chromosome02", 38567855), ("Chromosome03", 33309257),
    ("Chromosome04", 35938211), ("Chromosome05", 33681570), ("Chromosome06", 30819343),
    ("Chromosome07", 34668054), ("Chromosome08", 41806886), ("Chromosome09", 37838006),
    ("Chromosome10", 32070051), ("Chromosome11", 33098381), ("Chromosome12", 37136911),
    ("Chromosome13", 36952686), ("Chromosome14", 29308386), ("Chromosome15", 32923186),
    ("Chromosome16", 34171075), ("Chromosome17", 35156893), ("Chromosome18", 33477107),
]

# "within": hap1 x hap2 of the SAME accession -- a real heterozygous individual, not a
# synthetic cross. Picked accessions with clean two-haplotype assemblies in the index.
# "cross": different-accession pairs chosen by mash distance
# (cassava/relatedness/all_pairs_dist.tsv, built for this pipeline).
WITHIN_PAIRS = [
    ("TME204-hap1-v2", "TME204-hap2-v2"),
    ("COL386-hap1-v3", "COL386-hap2-v3"),
    ("TMe-3055-G9-hap1-v2", "TMe-3055-G9-hap2-v2"),
    ("IITA-TMS-IBA000070-hap1-v2", "IITA-TMS-IBA000070-hap2-v2"),
    ("BGM-2098-hap1-v2", "BGM-2098-hap2-v2"),
]

# Chosen by mash distance (cassava/relatedness/all_pairs_dist.tsv, built for this
# pipeline: `mash sketch` + `mash dist` over the 79 non-reference haplotype assemblies,
# same recipe as tripsacum/relatedness/). Same pattern as Tripsacum's PAIRS: a starter
# pair plus close matches sharing a member (TMEB693/TMe-2497-G5 cluster), plus one fully
# independent-member pair. Same-accession (hap1 x hap2) rows excluded here -- those are
# WITHIN_PAIRS above.
CROSS_PAIRS = [
    ("TMEB693-hap2-v2", "TMe-2497-G5-hap1-v2"),   # starter, dist=0.002714
    ("TMEB693-hap1-v2", "TMe-2497-G5-hap2-v2"),   # dist=0.002771 (opposite haps)
    ("TMEB693-hap2-v2", "CR63-hap1-v2"),          # dist=0.005510 (shares TMEB693-hap2)
    ("DSC493-12-1-hap2-v2", "COL40-DSC118-hap2-v2"),  # dist=0.003667 -- independent members
]

PAIRS = [("within", a, b) for a, b in WITHIN_PAIRS] + [("cross", a, b) for a, b in CROSS_PAIRS]

DEPTHS = [250_000]  # reads/haplotype; lowest depth in the prior NAM/Tripsacum sweeps scored best


def assembly_path(sample):
    p = ASM_DIR / f"{sample}.fa"
    if not p.exists():
        raise FileNotFoundError(f"no assembly for {sample!r}: {p}")
    return p


def combo_name(founders, depth):
    return f"{'x'.join(founders)}_{depth // 1000}k"


def simulate_reads(sample, depth, outdir):
    """wgsim-simulate `depth` read pairs directly from the assembly FASTA; keep R1 only.
    -r0/-R0 = no germline mutations/indels injected. Deterministic seed for cache/
    resumability (derived from Python's hash(sample) -- set PYTHONHASHSEED for exact
    cross-process reproducibility)."""
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
    selection, so its value here is irrelevant, only its presence/format matters."""
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
    """crf/ropebwt_npy_to_matrix.py at the true panel size (80) -- no --target-num-parents;
    the coverage-aware reduction to 24 that force-includes the true pair happens separately
    in select_to_24."""
    bins_path = outdir / "raw.npy.bins.tsv"
    gametes_path = outdir / "raw.npy.gametes.tsv"
    out_path = outdir / "windowed_k80.npy"

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
    emits, rather than hardcoding it -- safer with 80 hyphenated sample names than assuming
    a particular order matches the pangenome's internal founder-column order."""
    df = pd.read_csv(gametes_tsv, sep="\t")
    assert list(df.columns) == ["gameteIndex", "sampleName"], \
        f"unexpected gametes.tsv columns: {list(df.columns)}"
    assert len(df) == N_TARGET_PARENTS, \
        f"expected {N_TARGET_PARENTS} founders in gametes.tsv, got {len(df)}"
    return {row.sampleName: int(row.gameteIndex) for row in df.itertuples()}


def select_to_24(windowed_k80_path, idxA, idxB, outdir):
    """Reduce the true 80-founder panel down to the checkpoint's fixed 24-founder input,
    FORCE-INCLUDING the true pair (idxA, idxB) and filling the remaining 22 slots with the
    next most genome-wide-covered founders (coverage = fraction of rows with nonzero read
    support for that founder column) -- the same "keep what actually has support" logic
    `ropebwt_npy_to_matrix.py --target-num-parents` uses, except here the true pair is
    guaranteed to survive rather than merely likely to (it's supposed to be the two most-
    covered founders anyway, since all reads come from exactly those two samples; forcing
    it removes any risk from an unexpected coverage tie or cross-mapping surprise).
    Compacts label columns to the new 0..23 index space and writes diploid_k24.npy with
    both labels overwritten to the known-by-construction (idxA, idxB) pair -- same final
    effect as tripsacum_diploid.py's pad_to_24 + write_diploid_labels combined."""
    out_path = outdir / "diploid_k24.npy"
    if out_path.exists():
        return out_path, np.load(outdir / "select_to_24_keep.npy")

    arr = np.load(windowed_k80_path)
    N, T, C = arr.shape
    K80 = C - 2
    assert K80 == N_TARGET_PARENTS, f"expected {N_TARGET_PARENTS} founder cols, got {K80}"
    feats = arr[:, :, :K80]

    cov = (feats != 0).reshape(-1, K80).sum(axis=0)  # [K80] genome-wide coverage per founder
    rank = np.argsort(-cov, kind="stable")            # most-covered first
    rest = [int(i) for i in rank if i not in (idxA, idxB)][:N_MODEL_PARENTS - 2]
    keep = [int(idxA), int(idxB), *rest]
    assert len(keep) == N_MODEL_PARENTS

    a_rank = int(np.where(rank == idxA)[0][0])
    b_rank = int(np.where(rank == idxB)[0][0])
    print(f"  [{outdir.name}] select_to_24: true pair coverage rank a={a_rank} b={b_rank} "
          f"(0=most-covered of {K80})")

    out = np.zeros((N, T, N_MODEL_PARENTS + 2), dtype=arr.dtype)
    out[:, :, :N_MODEL_PARENTS] = feats[:, :, keep]
    out[:, :, N_MODEL_PARENTS] = 0      # idxA -> compacted index 0
    out[:, :, N_MODEL_PARENTS + 1] = 1  # idxB -> compacted index 1
    np.save(out_path, out)
    np.save(outdir / "select_to_24_keep.npy", np.array(keep))
    return out_path, np.array(keep)


def founder_density_stats(k80_npy_path, founder_idx):
    a = np.asarray(np.load(k80_npy_path, mmap_mode="r"))
    K = a.shape[-1] - 2
    feat = a[:, :, :K].reshape(-1, K)
    nz_row = feat != 0
    self_cov = nz_row[:, founder_idx].mean() * 100
    return self_cov


def evaluate_diploid_with_affinity(model, ds, device, batch_size, num_workers, ext_vec):
    """Same metric computation as crf/eval.py::evaluate_diploid, but broadcasts one
    genome-wide affinity vector (this run's own data, via _founder_affinity) to every
    window -- see tripsacum_diploid.py's identical function for full rationale."""
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
    tripsacum_diploid.py's run_diploid_eval. Auto-detects an affinity-trained checkpoint
    and computes/feeds its ext_emb accordingly; behaves exactly as before for the plain
    model."""
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


RESULT_COLS = ["kind", "assemblyA", "assemblyB", "depth_per_hap", "n_placed", "n_unplaced",
               "self_cov_A_pct", "self_cov_B_pct", "cov_rank_A", "cov_rank_B",
               "het_frac", "n_sites", "pair_acc", "hap_acc", "homo_pred"]


def write_header_if_needed(results_tsv=RESULTS_TSV):
    results_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not results_tsv.exists():
        results_tsv.write_text("\t".join(RESULT_COLS) + "\n")


def already_recorded(a, b, depth, results_tsv=RESULTS_TSV):
    """kind is a prefix column now, so the (a, b, depth) key -- tab-delimited on both
    sides to avoid partial-name collisions -- appears mid-line rather than at the start."""
    if not results_tsv.exists():
        return False
    key = f"\t{a}\t{b}\t{depth}\t"
    with open(results_tsv) as f:
        return any(key in line for line in f)


def run_one(kind, a, b, depth, device, force=False, ckpt_path=DIPLOID_CKPT, results_tsv=RESULTS_TSV):
    if not force and already_recorded(a, b, depth, results_tsv):
        print(f"[{a}x{b}@{depth}] already in results TSV, skipping entirely")
        return

    name = combo_name([a, b], depth)
    outdir = SCRATCH / name
    print(f"\n=== {name} ({kind}) ===")

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
        row = dict(kind=kind, assemblyA=a, assemblyB=b, depth_per_hap=depth,
                    n_placed=n_placed, n_unplaced=n_unplaced,
                    self_cov_A_pct=float("nan"), self_cov_B_pct=float("nan"),
                    cov_rank_A=-1, cov_rank_B=-1,
                    het_frac=float("nan"), n_sites=0, pair_acc=float("nan"),
                    hap_acc=float("nan"), homo_pred=float("nan"))
    else:
        k80_npy, gametes_tsv = window(raw_npy, outdir)
        panel = discover_panel_order(gametes_tsv)
        if a not in panel or b not in panel:
            raise RuntimeError(f"{name}: {a!r}/{b!r} not found in discovered panel "
                                f"{sorted(panel)}")
        idxA, idxB = panel[a], panel[b]

        self_cov_A = founder_density_stats(k80_npy, idxA)
        self_cov_B = founder_density_stats(k80_npy, idxB)

        diploid_npy, keep = select_to_24(k80_npy, idxA, idxB, outdir)
        cov_rank_A = int(np.where(keep == idxA)[0][0])  # always 0 by construction
        cov_rank_B = int(np.where(keep == idxB)[0][0])  # always 1 by construction
        assert idxA in keep and idxB in keep, \
            f"{name}: true pair not in selected 24-founder panel -- select_to_24 bug"

        arr = np.load(diploid_npy, mmap_mode="r")
        K = arr.shape[-1] - 2
        het_frac = float((arr[:, :, K] != arr[:, :, K + 1]).mean())
        if het_frac != 1.0:
            raise RuntimeError(f"{name}: expected 100% het by construction, got "
                                f"{het_frac * 100:.2f}% -- label overwrite bug")

        r = run_diploid_eval(diploid_npy, name, device, ckpt_path=ckpt_path)
        row = dict(kind=kind, assemblyA=a, assemblyB=b, depth_per_hap=depth,
                    n_placed=n_placed, n_unplaced=n_unplaced,
                    self_cov_A_pct=self_cov_A, self_cov_B_pct=self_cov_B,
                    cov_rank_A=cov_rank_A, cov_rank_B=cov_rank_B,
                    het_frac=het_frac, n_sites=r["n"], pair_acc=r["pair_acc"],
                    hap_acc=r["hap_acc"], homo_pred=r["homo_pred"])

    write_header_if_needed(results_tsv)
    if force and already_recorded(a, b, depth, results_tsv):
        lines = results_tsv.read_text().splitlines(keepends=True)
        key = f"\t{a}\t{b}\t{depth}\t"
        keep_lines = [l for l in lines if key not in l]
        results_tsv.write_text("".join(keep_lines))
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
    df = df.sort_values(["kind", "assemblyA", "assemblyB", "depth_per_hap"]).reset_index(drop=True)
    float_cols = {"self_cov_A_pct", "self_cov_B_pct", "het_frac", "pair_acc",
                  "hap_acc", "homo_pred"}
    table_md = _markdown_table(df, float_cols)

    model_desc = ("the **affinity-conditioned** (`founder_affinity=True`, "
                   "`checkpoints/diploid-affinity-sim512-h3`) diploid GRITS-CRF, fed a "
                   "genome-wide `_founder_affinity` vector computed from this run's own "
                   "data"
                   if affinity else
                   "the plain (non-affinity; `founder_affinity=False`) diploid GRITS-CRF "
                   "via `crf/eval.py::evaluate_diploid`")
    lines = [
        f"# Synthetic-diploid cassava baseline (simulated reads, "
        f"{'affinity' if affinity else 'plain'} diploid CRF)\n",
        "Each row simulates reads directly from two assembly FASTAs (`wgsim -e 0.001 -r 0 "
        "-R 0 -1 100 -2 100`, R1 only kept), combines them into one FASTQ with no "
        "recombination (truth = `(assemblyA, assemblyB)` at every site by construction), "
        "runs the combined reads through `ropebwt3 refmap` against the cassava 80-founder "
        "pangenome (`cassavaChrIndex.{fmd,lift}`, `--ref-prefix="
        f"{REF_PREFIX}`), windows at the true panel size K=80 "
        "(`crf/ropebwt_npy_to_matrix.py --window-size=512`), then SELECTS DOWN to K=24 "
        "(true pair force-included + the 22 next most genome-wide-covered founders) to "
        "match the checkpoint's fixed 24-founder architecture -- the reverse of Tripsacum's "
        "18->24 pad-up, since cassava's indexed panel (80) is larger than the model. Scored "
        f"with {model_desc}.\n\n"
        "`kind=within` rows pair the two haplotypes of the SAME accession (a genuinely real "
        "heterozygous individual, unique to cassava's hap-resolved assemblies among our "
        "datasets); `kind=cross` rows pair different accessions chosen by mash distance "
        "(`cassava/relatedness/all_pairs_dist.tsv`).\n\n"
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
        needed = {a for _, a, b in PAIRS} | {b for _, a, b in PAIRS}
        for name in sorted(needed):
            try:
                assembly_path(name)
                status = "OK"
            except FileNotFoundError:
                status = "MISSING"
            print(f"{name:<30} {status}")
        print(f"\nPairs ({len(PAIRS)}): {PAIRS}")
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
            raise SystemExit(f"usage: cassava_diploid.py {mode} <A> <B> <depth>")
        a, b, depth = sys.argv[2], sys.argv[3], int(sys.argv[4])
        kind = "within" if any(a == x and b == y for x, y in WITHIN_PAIRS) else "cross"
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = AFFINITY_CKPT if mode == "one-affinity" else DIPLOID_CKPT
        tsv = AFFINITY_RESULTS_TSV if mode == "one-affinity" else RESULTS_TSV
        write_header_if_needed(tsv)
        run_one(kind, a, b, depth, device, force=True, ckpt_path=ckpt, results_tsv=tsv)
    elif mode in ("all", "all-affinity"):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = AFFINITY_CKPT if mode == "all-affinity" else DIPLOID_CKPT
        tsv = AFFINITY_RESULTS_TSV if mode == "all-affinity" else RESULTS_TSV
        write_header_if_needed(tsv)
        for kind, a, b in PAIRS:
            for depth in DEPTHS:
                run_one(kind, a, b, depth, device, ckpt_path=ckpt, results_tsv=tsv)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    print(f"\nDone. Results in {AFFINITY_RESULTS_TSV if 'affinity' in mode else RESULTS_TSV}")


if __name__ == "__main__":
    main()
