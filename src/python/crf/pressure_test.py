"""
Pressure test for the GRITS-CRF model.

Checks GPU availability, data loading, forward/backward pass correctness,
label validity, memory use, and throughput.  All outputs go to --workdir.

Usage:
    pixi run --environment gpu python src/python/crf/pressure_test.py
    pixi run --environment gpu python src/python/crf/pressure_test.py \
        --workdir /workdir/esb33 --data-dir /workdir/smm477/ML-data/training \
        --batch-size 16 --steps 50
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_crf import GRITSCRFModel, LabeledDatasetDiploid, create_diploid_pairs

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return condition


def require(name, condition, detail=""):
    ok = check(name, condition, detail)
    if not ok:
        print(f"\n  Cannot continue without: {name}")
        sys.exit(1)
    return ok


# --------------------------------------------------------------------------- #
#  Test functions                                                               #
# --------------------------------------------------------------------------- #

def test_environment():
    section("Environment")
    check("Python version >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])

    import torch
    check("torch importable", True, torch.__version__)
    cuda_ok = torch.cuda.is_available()
    require("CUDA available", cuda_ok,
            "Run with: pixi run --environment gpu python ...")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    check("GPU detected", True, f"{gpu_name}  ({vram_gb:.1f} GB VRAM)")

    import pytorch_lightning as pl
    check("pytorch_lightning importable", True, pl.__version__)

    return torch.device("cuda")


def test_data(data_dir, num_parents, window_size, step_size):
    section("Data loading")

    train_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    require("Training files found", len(train_files) > 0,
            f"looked in {data_dir}")
    print(f"  {INFO}  {len(train_files)} .npy files found")

    # Use a single small file for speed
    test_file = train_files[0]
    print(f"  {INFO}  Loading: {test_file}")

    mat = np.load(os.path.join(data_dir, test_file), allow_pickle=True, mmap_mode='r')
    check("Matrix loads", True, f"shape={mat.shape}  dtype={mat.dtype}")

    n_cols_expected = num_parents + 2
    check("Column count matches num_parents + 2 labels",
          mat.shape[1] == n_cols_expected,
          f"got {mat.shape[1]}, expected {n_cols_expected}")

    check("Values are non-negative", int(mat.min()) >= 0,
          f"min={mat.min()}")

    # Label range check
    labels_col1 = mat[:, num_parents]
    labels_col2 = mat[:, num_parents + 1]
    max_label = max(int(labels_col1.max()), int(labels_col2.max()))
    pairs, _ = create_diploid_pairs(num_parents)
    n_pair_states = len(pairs)

    check("Label values within [0, num_parents-1]",
          max_label < num_parents,
          f"max label={max_label}, num_parents={num_parents}")

    # Dataset
    ds = LabeledDatasetDiploid(data_dir, [test_file],
                               window_size=window_size,
                               num_parents=num_parents,
                               step_size=step_size)
    check("Dataset created", len(ds) > 0, f"{len(ds)} windows")

    sample = ds[0]
    check("Sample input shape",
          sample["input_embeds"].shape == (window_size, num_parents),
          str(sample["input_embeds"].shape))
    check("Sample labels shape",
          sample["labels"].shape == (window_size,),
          str(sample["labels"].shape))

    max_pair_idx = n_pair_states - 1
    label_max = int(sample["labels"].max())
    check("Sample label indices within pair-state range",
          label_max <= max_pair_idx,
          f"max label idx={label_max}, n_pair_states={n_pair_states}")

    return ds


def test_model_forward(device, num_parents, window_size, d_model, n_heads, n_layers):
    section("Model forward pass")

    model = GRITSCRFModel(num_parents=num_parents, d_model=d_model,
                          n_heads=n_heads, n_layers=n_layers)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check("Model created", True, f"{n_params:,} trainable params")

    model = model.to(device)
    check("Model moved to GPU", next(model.parameters()).device.type == "cuda")

    B, T, K = 2, window_size, num_parents
    X = torch.rand(B, T, K, device=device)
    tags = torch.randint(0, model.pair_idx.shape[0], (B, T), device=device)

    try:
        emis_p, tr, g, c = model(X)
        check("Forward pass", True,
              f"emis_p={tuple(emis_p.shape)}  tr={tuple(tr.shape)}")
        check("Gate values in (0,1)", bool((g > 0).all() and (g < 1).all()),
              f"min={g.min():.3f}  max={g.max():.3f}")
        check("Recomb c positive", bool((c > 0).all()),
              f"min={c.min():.4f}  max={c.max():.4f}")
    except Exception as e:
        check("Forward pass", False, str(e))
        traceback.print_exc()
        return None, None

    try:
        loss = model.crf.nll(emis_p, tr, tags)
        check("CRF NLL finite", bool(torch.isfinite(loss)),
              f"loss={loss.item():.4f}")
    except Exception as e:
        check("CRF NLL", False, str(e))
        return None, None

    return model, (emis_p, tr, g, c)


def test_backward(model, device, num_parents, window_size):
    section("Backward pass & gradients")

    B, T, K = 2, window_size, num_parents
    X = torch.rand(B, T, K, device=device)
    tags = torch.randint(0, model.pair_idx.shape[0], (B, T), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.zero_grad()

    try:
        emis_p, tr, g, c = model(X)
        gate_loss = 0.05 * (1.0 - g).mean()
        loss = model.crf.nll(emis_p, tr, tags) + gate_loss
        loss.backward()
        check("Backward pass", True, f"loss={loss.item():.4f}")
    except Exception as e:
        check("Backward pass", False, str(e))
        traceback.print_exc()
        return False

    grad_norms = [p.grad.norm().item() for p in model.parameters()
                  if p.grad is not None]
    any_nonzero = any(g > 0 for g in grad_norms)
    check("Gradients non-zero", any_nonzero,
          f"max grad norm={max(grad_norms):.4f}")

    optimizer.step()
    check("Optimizer step", True)

    return True


def test_throughput(model, ds, device, batch_size, steps, workdir):
    section(f"Throughput  (batch={batch_size}, steps={steps})")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    losses = []

    for step, batch in enumerate(loader):
        if step >= steps:
            break
        X = batch["input_embeds"].to(device)
        tags = batch["labels"].to(device)

        optimizer.zero_grad()
        emis_p, tr, g, c = model(X)
        loss = model.crf.nll(emis_p, tr, tags) + 0.05 * (1.0 - g).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 10 == 0:
            print(f"    step {step+1}/{steps}  loss={loss.item():.4f}")

    elapsed = time.perf_counter() - t0
    actual_steps = len(losses)
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    check("All steps completed", actual_steps == steps,
          f"{actual_steps}/{steps}")
    check("Loss finite throughout", all(np.isfinite(losses)),
          f"first={losses[0]:.2f}  last={losses[-1]:.2f}")

    its = actual_steps / elapsed
    print(f"  {INFO}  {its:.1f} it/s  |  peak VRAM={peak_mem_gb:.2f} GB  |  "
          f"elapsed={elapsed:.1f}s")

    # Write results
    result_path = Path(workdir) / "results" / "pressure_test.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        f.write(f"batch_size={batch_size}\n")
        f.write(f"steps={actual_steps}\n")
        f.write(f"throughput_its={its:.2f}\n")
        f.write(f"peak_vram_gb={peak_mem_gb:.2f}\n")
        f.write(f"initial_loss={losses[0]:.4f}\n")
        f.write(f"final_loss={losses[-1]:.4f}\n")
    print(f"  {INFO}  Results written to {result_path}")

    return its, peak_mem_gb


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="GRITS-CRF pressure test")
    p.add_argument("--workdir", default="/workdir/esb33",
                   help="Root output directory (default: /workdir/esb33)")
    p.add_argument("--data-dir", default="/workdir/smm477/ML-data/training",
                   help="Directory of .npy training matrices")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--window-size", type=int, default=512)
    p.add_argument("--step-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=20,
                   help="Training steps for throughput test")
    return p.parse_args()


def main():
    args = parse_args()

    print("\nGRITS-CRF Pressure Test")
    print(f"workdir : {args.workdir}")
    print(f"data-dir: {args.data_dir}")

    all_pass = True

    device = test_environment()

    ds = test_data(args.data_dir, args.num_parents,
                   args.window_size, args.step_size)

    model, _ = test_model_forward(device, args.num_parents, args.window_size,
                                  args.d_model, args.n_heads, args.n_layers)
    if model is None:
        all_pass = False
    else:
        ok = test_backward(model, device, args.num_parents, args.window_size)
        all_pass = all_pass and ok

        if ok:
            test_throughput(model, ds, device, args.batch_size,
                            args.steps, args.workdir)

    section("Summary")
    if all_pass:
        print(f"  {PASS}  All checks passed — ready to train.\n")
    else:
        print(f"  {FAIL}  Some checks failed — review output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
