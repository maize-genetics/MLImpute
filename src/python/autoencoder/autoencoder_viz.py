import torch
import matplotlib.pyplot as plt

from autoencoder_model import AutoEncoder, AutoEncoder2, AutoEncoder3


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

    # Slice the desired window: [B, num_parents, window_size]
    window = matrix[:, :, window_start:window_start + window_size]

    # Permute to [B, window_size, num_parents] for the model forward
    inp = window.permute(0, 2, 1).to(device)  # [B, 512, 25]

    # Run through model
    with torch.no_grad():
        out = model(inp).cpu().squeeze(0)  # [num_parents, window_size]

    out = (out >= 0.5).float()

    # Convert to numpy for plotting
    inp_np = inp.squeeze(0).cpu().numpy().T
    out_np = out.numpy().T

    # Plot input vs output
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    im1 = axes[0].imshow(inp_np, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[0].set_title("Noisy Input")
    axes[0].set_ylabel("Parents")
    axes[0].set_xlabel("Genomic Position")

    im2 = axes[1].imshow(out_np, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[1].set_title("Denoised Output")
    axes[1].set_xlabel("Genomic Position")

    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # high-res PNG
    plt.close(fig)  # close so it doesn?t display inline


save_path = "/workdir/zrm22/testPixi/MLImpute/images/autoencoder_hd64_bd16_ChatGPTSuggest_ep3_1k.png"
# Example usage
matrix = torch.randint(0, 2, (25, 5000)).float()  # fake binary matrix [25 parents x 5000 positions]

# Step 1: Recreate model
model = AutoEncoder3(num_parents=25, window_size=512, hidden_dim=64, bottleneck_dim=16, dropout=0.1)

# Step 2: Load weights
state_dict = torch.load("/workdir/zrm22/testPixi/MLImpute/saved_models/autoencoder_hd64_bd16_ChatGPTSuggest/3.pth")
model.load_state_dict(state_dict)

# Step 3: Switch to eval mode
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# Pick a region (say positions 1000?1512)
visualize_denoising(model, matrix, save_path, window_start=1000, window_size=512)

