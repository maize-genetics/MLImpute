import torch.nn as nn



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