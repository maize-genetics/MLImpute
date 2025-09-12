import torch
import matplotlib.pyplot as plt
import os
import argparse
from encoder_decoder import Encoder, Decoder, Seq2Seq, WindowIndexDataset
from pathlib import Path

def gather_npy_paths(root_dir):
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(root_dir)
        for f in files if f.endswith(".npy")
    ]


def visualize_denoising(model, matrix, save_path, window_start=0, window_size=512, device="cuda"):
    """
    matrix: torch.Tensor of shape [num_parents, total_window_size]
    model: trained autoencoder (expects [B, C, L])
    window_start: start index of the window to visualize
    window_size: number of positions in the window
    """

    if matrix.dim() == 2:  # [num_parents, total_window_size]
        matrix = matrix.unsqueeze(0)  # [1, num_parents, total_window_size]
    elif matrix.dim() == 3:
        pass  # already [B, num_parents, total_window_size]
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {matrix.shape}")

    print(matrix.shape)
    # Slice the desired window: [B, num_parents, window_size]
    window = matrix[:, :, window_start:window_start + window_size]

    print(window.shape)
    # Permute to [B, window_size, num_parents] for the model forward
    inp = window.permute(0, 2, 1).to(device)  # [B, 512, 25]

    print(inp.shape)

    # Run through model
    with torch.no_grad():
        out = model(inp).cpu().squeeze(0)  # [num_parents, window_size]

    out = (out >= 1.0).float()

    # Convert to numpy for plotting
    inp_np = inp.squeeze(0).cpu().numpy().T
    out_np = out.numpy().T

    # Plot input vs output stacked vertically
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    im1 = axes[0].imshow(inp_np, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[0].set_title("Noisy Input")
    axes[0].set_ylabel("Parents")

    im2 = axes[1].imshow(out_np, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[1].set_title(f"Denoised Output (threshold=0)")
    axes[1].set_xlabel("Genomic Position")
    axes[1].set_ylabel("Parents")

    # Add colorbars
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # high-res PNG
    plt.close(fig)  # close so it doesn?t display inline


def plot_output_distributions(model, matrix, window_start=0, window_size=512,
                              device="cuda", bins=50, save_path=None):
    """
    Plot the distributions of autoencoder outputs grouped by input=1 vs input=0.
    """
    # Ensure input is [B, num_parents, total_window_size]
    if matrix.dim() == 2:  # [num_parents, total_window_size]
        matrix = matrix.unsqueeze(0)
    elif matrix.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got {matrix.shape}")

    # Slice window
    window = matrix[:, :, window_start:window_start + window_size]  # [B, P, W]

    # Prepare for model
    inp = window.permute(0, 2, 1).to(device)  # [B, W, P]

    # Forward pass
    with torch.no_grad():
        out = model(inp).cpu()  # [B, W, P]

    # Flatten
    inp_flat = inp.cpu().numpy().flatten()
    out_flat = out.numpy().flatten()

    # Group outputs by whether input was 1 or 0
    out_when_1 = out_flat[inp_flat == 1]
    out_when_0 = out_flat[inp_flat == 0]

    # Plot distributions
    plt.figure(figsize=(10, 6))
    plt.hist(out_when_0, bins=bins, alpha=0.6, label="Input = 0", density=True)
    plt.hist(out_when_1, bins=bins, alpha=0.6, label="Input = 1", density=True)
    plt.xlabel("Autoencoder Output Value")
    plt.ylabel("Density")
    plt.title("Output Distribution Grouped by Input Value")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Haplotype Imputation Tool")
    parser.add_argument("--model", "-m", type=Path, required=True, help="Path to input file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Path to output plots.")
    parser.add_argument("--window-size", type=int, default=512, help="Window size for the autoencoder")
    parser.add_argument("--num-classes", type=int, default=25, help="Number of parental haplotypes (classes)")

    args = parser.parse_args()

    save_path = str(args.output) + "_matrix.png"
    save_path2 = str(args.output) + "_distributions.png"
    # Example usage

    test_paths = gather_npy_paths("training_data/justCML69")

    test_dataset = WindowIndexDataset(test_paths, window_size=args.window_size, top_n=args.num_classes,
                                      step_size=args.window_size, return_decode=True)
    matrix, decode = test_dataset[1000]

    matrix = matrix.unsqueeze(0).permute(0, 2, 1)
    print(matrix.shape)

    # Step 1: Recreate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(emb_dim=25, hid_dim=512, n_layers=3, dropout=0.5, device=device)
    decoder = Decoder(output_dim=25, emb_dim=512, hid_dim=512, n_layers=6, dropout=0.5)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device)

    # model = UNet1D(num_parents=25, hidden_dim=128, bottleneck_dim=50, dropout=0.1)

    # Step 2: Load weights

    state_dict = torch.load(args.model)

    model.load_state_dict(state_dict)

    # Step 3: Switch to eval mode
    model.eval()

    model.to(device)

    visualize_denoising(model, matrix, save_path, window_start=0, window_size=512)

    plot_output_distributions(model, matrix, window_start=0, window_size=512, save_path=save_path2)

if __name__ == "__main__":
    main()