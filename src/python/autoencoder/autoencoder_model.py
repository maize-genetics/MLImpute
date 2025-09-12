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

