import torch.nn as nn
import torch



class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, targets):
        return self.loss_fct(logits, targets)


class AutoEncoder(nn.Module):
    def __init__(self, num_parents, window_size, hidden_dim, bottleneck_dim, dropout):
        super().__init__()
        self.parents = num_parents
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.loss = BCELoss()

        # Encoder: 1 -> hidden -> bottleneck
        #narrow 2d alternating with bigger 1d on position dimension
        # Idea is that the 2d spans all gametes and a set of positions and will allow the model to learn
        self.enc = nn.Sequential(
            nn.Conv1d(self.window_size, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, bottleneck_dim, kernel_size=1),
            nn.GELU(),
        )

        # Decoder: bottleneck -> hidden -> 1
        self.dec = nn.Sequential(
            nn.Conv1d(bottleneck_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, self.window_size, kernel_size=1),
        )

    def forward(self, x):
        # input: (batch_size, window_size, num_parents)
        # output: (batch_size, window_size, num_parents)
        #x = x.permute(0, 2, 1)
        encoded = self.enc(x)
        decoded = self.dec(encoded)
        #return decoded.permute(0, 2, 1)
        return decoded


    def compute_loss(self, logits, unmasked):
        return self.loss(logits, unmasked)

class AutoEncoder2(nn.Module):
    def __init__(self, num_parents, window_size, hidden_dim, bottleneck_dim, dropout):
        super().__init__()
        self.parents = num_parents
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.loss = BCELoss()

        # Encoder: 1 -> hidden -> bottleneck
        #narrow 2d alternating with bigger 1d on position dimension
        # Idea is that the 2d spans all gametes and a set of positions and will allow the model to learn
        self.enc = nn.Sequential(
            nn.Conv1d(self.window_size, hidden_dim, kernel_size=20),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, bottleneck_dim, kernel_size=20),
            nn.GELU(),
        )

        # Decoder: bottleneck -> hidden -> 1
        self.dec = nn.Sequential(
            nn.Conv1d(bottleneck_dim, hidden_dim, kernel_size=20),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, self.window_size, kernel_size=20),
        )

    def forward(self, x):
        # input: (batch_size, window_size, num_parents)
        # output: (batch_size, window_size, num_parents)
        x = x.permute(0, 2, 1)
        encoded = self.enc(x)
        decoded = self.dec(encoded)
        return decoded.permute(0, 2, 1)
        return decoded


    def compute_loss(self, logits, unmasked):
        return self.loss(logits, unmasked)


##This one was suggested by chat GPT
class AutoEncoder3(nn.Module):
    def __init__(self, num_parents, window_size, hidden_dim, bottleneck_dim, dropout):
        super().__init__()
        self.parents = num_parents
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.loss = BCELoss()

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv1d(num_parents, hidden_dim, kernel_size=5, padding=2),  # context of 5
            nn.GELU(),
            nn.Conv1d(hidden_dim, bottleneck_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.Conv1d(bottleneck_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, num_parents, kernel_size=5, padding=2)
        )

    def forward(self, x):
        # input: (batch_size, window_size, num_parents)

        # output: (batch_size, window_size, num_parents)
        x = x.permute(0, 2, 1)
        encoded = self.enc(x)
        decoded = self.dec(encoded)
        return decoded.permute(0, 2, 1)
        # return decoded

    def compute_loss(self, logits, unmasked):
        return self.loss(logits, unmasked)


class ConvBlock(nn.Module):
    """Two Conv1d + GELU + optional Dropout"""

    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class UNet1D(nn.Module):
    def __init__(self, num_parents=25, hidden_dim=64, bottleneck_dim=128, dropout=0.1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(num_parents, hidden_dim, dropout)  # [B, 64, 512]
        self.down1 = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)  # [B, 64, 256]

        self.enc2 = ConvBlock(hidden_dim, bottleneck_dim, dropout)  # [B, 128, 256]
        self.down2 = nn.Conv1d(bottleneck_dim, bottleneck_dim, 4, stride=2, padding=1)  # [B, 128, 128]

        # Bottleneck
        self.bottleneck = ConvBlock(bottleneck_dim, bottleneck_dim, dropout)  # [B, 128, 128]

        # Decoder
        self.up2 = nn.ConvTranspose1d(bottleneck_dim, bottleneck_dim, 4, stride=2, padding=1)  # [B, 128, 256]
        self.dec2 = ConvBlock(bottleneck_dim * 2, hidden_dim, dropout)  # skip from enc2

        self.up1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)  # [B, 64, 512]
        self.dec1 = ConvBlock(hidden_dim * 2, hidden_dim, dropout)  # skip from enc1

        # Final prediction
        self.out_conv = nn.Conv1d(hidden_dim, num_parents, kernel_size=1)  # [B, 25, 512]
        self.out_act = nn.Sigmoid()  # probability of support

        self.loss = BCELoss()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # Encoder
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        # Bottleneck
        b = self.bottleneck(d2)

        # Decoder
        u2 = self.up2(b)
        cat2 = torch.cat([u2, e2], dim=1)  # skip connection
        d2_out = self.dec2(cat2)

        u1 = self.up1(d2_out)
        cat1 = torch.cat([u1, e1], dim=1)  # skip connection
        d1_out = self.dec1(cat1)

        out = self.out_conv(d1_out)
        out = out.permute(0, 2, 1)
        return out

    def compute_loss(self, logits, unmasked):
        return self.loss(logits, unmasked)



class AutoEncoder4(nn.Module):
    def __init__(self, num_parents, window_size, hidden_dim, bottleneck_dim, dropout):
        super().__init__()
        self.parents = num_parents
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.loss = BCELoss()

        # Encoder_conv
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(num_parents, num_parents, kernel_size=5, padding=2),  # context of 5
            nn.GELU(),
            nn.Conv1d(num_parents, num_parents, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Flatten + FC bottleneck
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_enc = nn.Sequential(
            nn.LazyLinear(hidden_dim),  # learns input size automatically
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),  # latent bottleneck
        )

        # ----- Decoder -----
        self.fc_dec = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.GELU(),
            nn.GELU(),
            nn.LazyLinear(num_parents * window_size),  # unflatten to [B, 64, ?] (will fix size dynamically)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(num_parents, window_size))

        self.decoder_conv = nn.Sequential(
            nn.LazyConvTranspose1d(num_parents, kernel_size=5, padding=2),  # upsample
            nn.GELU(),
            nn.LazyConvTranspose1d(num_parents, kernel_size=5, padding=2),  # back to 25 channels
        )

    def forward(self, x):
        # input: (batch_size, window_size, num_parents)

        # output: (batch_size, window_size, num_parents)
        x = x.permute(0, 2, 1)
        # Encoder
        x = self.encoder_conv(x)
        x = self.flatten(x)
        z = self.fc_enc(x)

        # Decoder
        x = self.fc_dec(z)
        x = self.unflatten(x)  # reshape back to [B, 25, 512]
        x = self.decoder_conv(x)
        return x.permute(0, 2, 1)
        # return decoded

    def compute_loss(self, logits, unmasked):
        return self.loss(logits, unmasked)



# class ParentConv(nn.Module):
#     def __init__(self, num_parents=25, embedding_dim=16):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.conv1 = nn.Conv1d(
#             in_channels=num_parents,
#             out_channels=num_parents * embedding_dim,
#             kernel_size=5,
#             padding=2,
#             groups=num_parents  # depthwise
#         )
#         self.act1 = nn.GELU()
#         self.conv2 = nn.Conv1d(
#             in_channels=num_parents * embedding_dim,
#             out_channels=num_parents * embedding_dim,
#             kernel_size=5,
#             padding=2,
#             groups=num_parents  # keep parents separate
#         )
#         self.act2 = nn.GELU()
#
#     def forward(self, x):
#         # x: [B, num_parents, 512]
#         x = self.act1(self.conv1(x))
#         x = self.act2(self.conv2(x))
#         # reshape to [B, num_parents, embedding_dim, 512]
#         B, C, L = x.shape
#         x = x.view(B, -1, self.embedding_dim, L)
#         return x
#
# class ParentAttention(nn.Module):
#     def __init__(self, num_parents=25, embedding_dim=16, seq_len=512, num_heads=4):
#         super().__init__()
#         self.embed_dim = embedding_dim * seq_len
#         self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)
#
#     def forward(self, x):
#         # x: [B, num_parents, embedding_dim, 512]
#         B, N, D, L = x.shape
#         x = x.permute(0, 1, 3, 2).reshape(B, N, D * L)  # [B, num_parents, embed_dim*L]
#         out, _ = self.attn(x, x, x)  # attention over parents
#         out = out.reshape(B, N, L, D).permute(0, 1, 3, 2)  # back to [B, num_parents, embedding_dim, 512]
#         return out
#
#
# class ParentAutoencoder(nn.Module):
#     def __init__(self, num_parents=25, window_size = 512, embedding_dim=16, latent_dim=64):
#         super().__init__()
#         self.encoder_conv = ParentConv(num_parents=num_parents, embedding_dim=embedding_dim)
#         self.parent_attention = ParentAttention(num_parents=num_parents, embedding_dim=embedding_dim)
#
#         # compress across positions (flatten embedding+position, then linear down)
#         self.to_latent = nn.Linear(embedding_dim * window_size, latent_dim)
#         self.from_latent = nn.Linear(latent_dim, embedding_dim * window_size)
#
#         self.decoder_conv = ParentConv(num_parents=num_parents, embedding_dim=embedding_dim)
#         self.loss = BCELoss()
#
#     def forward(self, x):
#         # Encoder
#         x = self.encoder_conv(x)  # [B, N, D, window_size]
#         x = self.parent_attention(x)  # attention across parents
#         B, N, D, L = x.shape
#         x = x.view(B, N, D * L)
#         x = self.to_latent(x)  # compress
#
#         # Decoder
#         x = self.from_latent(x)  # decompress
#         x = x.view(B, N, D, L)
#         x = self.decoder_conv(
#             x.view(B, N * D, L)
#         )  # reconstruct
#         x = x.view(B, N, D, L)
#         x = x.mean(dim=2)
#         return x
#
#     def compute_loss(self, logits, unmasked):
#         return self.loss(logits, unmasked)


class ParentConv(nn.Module):
    """
    Depthwise Conv1d over parents with multi-dimensional embeddings.
    Each parent has its own embedding_dim channels, and convolution
    is applied independently per parent.
    """

    def __init__(self, num_parents=25, embedding_dim=16, kernel_size=5):
        super().__init__()
        self.num_parents = num_parents
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size

        # Each parent's group has embedding_dim channels
        self.conv1 = nn.Conv1d(
            in_channels=num_parents * embedding_dim,
            out_channels=num_parents * embedding_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=num_parents  # depthwise
        )
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            in_channels=num_parents * embedding_dim,
            out_channels=num_parents * embedding_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=num_parents  # depthwise
        )
        self.act2 = nn.GELU()

    def forward(self, x):
        """
        x: [B, num_parents, embedding_dim, seq_len]
        Returns: [B, num_parents, embedding_dim, seq_len]
        """
        B, N, D, L = x.shape
        assert N == self.num_parents and D == self.embedding_dim, \
            f"Expected input [B, {self.num_parents}, {self.embedding_dim}, L], got {x.shape}"

        # Merge parent and embedding_dim for depthwise Conv1d
        x = x.view(B, N * D, L)           # [B, num_parents*embedding_dim, seq_len]

        # Conv1d layers
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))

        # Reshape back to [B, num_parents, embedding_dim, seq_len]
        x = x.view(B, N, D, L)
        return x
# -------------------------------
# Parent Attention Module
# -------------------------------
class ParentAttention(nn.Module):
    """
    Multi-head attention across parents, per position.
    Input: [B, num_parents, embedding_dim, seq_len]
    Output: same shape
    """
    def __init__(self, num_parents=25, embedding_dim=16, num_heads=4):
        super().__init__()
        self.num_parents = num_parents
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Attention embedding_dim per parent, will flatten seq_len into features
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, N, D, L = x.shape
        # Attention per position: treat embedding_dim as features per parent
        # We'll attend across parents for each position independently
        # First, reshape: combine batch*seq_len for convenience
        x = x.permute(0, 3, 1, 2).reshape(B * L, N, D)  # [B*seq_len, num_parents, embedding_dim]
        out, _ = self.attn(x, x, x)
        out = out.reshape(B, L, N, D).permute(0, 2, 3, 1)  # [B, num_parents, embedding_dim, seq_len]
        return out

# -------------------------------
# Full Autoencoder
# -------------------------------
class ParentAutoencoder(nn.Module):
    def __init__(self, num_parents=25, embedding_dim=16, latent_dim=64):
        super().__init__()
        self.num_parents = num_parents
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.loss = BCELoss()

        # Encoder conv
        self.encoder_conv = ParentConv(num_parents=num_parents, embedding_dim=embedding_dim)

        # Attention across parents
        self.parent_attention = ParentAttention(num_parents=num_parents, embedding_dim=embedding_dim)

        # Bottleneck: compress each parent’s embedding over positions
        self.to_latent = nn.Linear(embedding_dim * 512, latent_dim)
        self.from_latent = nn.Linear(latent_dim, embedding_dim * 512)

        # Decoder conv (same as encoder)
        self.decoder_conv = ParentConv(num_parents=num_parents, embedding_dim=embedding_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(2).repeat(1, 1, 16, 1)  # [B, num_parents, embedding_dim, seq_len]

        # x: [B, num_parents, embedding_dim, seq_len]
        x = self.encoder_conv(x)
        x = self.parent_attention(x)

        B, N, D, L = x.shape
        # Flatten positions + embedding for latent
        x = x.reshape(B, N, D * L)
        x = self.to_latent(x)           # [B, N, latent_dim]
        x = self.from_latent(x)         # [B, N, D*L]
        x = x.reshape(B, N, D, L)      # [B, num_parents, embedding_dim, seq_len]

        # Decoder convolution: outputs logits (no activation)
        x = self.decoder_conv(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        return x  # [B, num_parents, embedding_dim, seq_len]

    def compute_loss(self, logits, unmasked):
        return self.loss(logits, unmasked)





