# modified from https://medium.com/@challamalla5sunil/building-a-machine-translation-system-with-encoder-decoder-architecture-in-pytorch-b7373fee1760

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import numba
from typing import Optional, Tuple, Union
from transformers import ModernBertConfig, ModernBertModel

class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        #self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, src):
        # src dimension (batch size, src len) - this is 2d tensor because it only has token indices.
        #embedded = self.dropout(self.embedding(src))
        embedded = self.dropout(src)
        # embedded dimension (batch size, src len, emb dim)
        batch_size = src.shape[0]
        hidden = torch.zeros(2 * self.n_layers, batch_size, self.hid_dim).to(self.device)
        cell = torch.zeros(2 * self.n_layers, batch_size, self.hid_dim).to(self.device)
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # outputs dimension (batch size, src len, hid dim * n directions)
        # hidden dimension (n layers * n directions, batch size, hid dim)
        # cell dimension (n layers * n directions, batch size, hid dim)
        return hidden, cell


class ModernBERTEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len, h_out, h_cell, n_layers):
        super().__init__()

        config = ModernBertConfig(max_position_embeddings=seq_len, hidden_size=vocab_size, num_attention_heads=vocab_size)

        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.bert = ModernBertModel(config)
        self.h_out = h_out
        self.h_cell = h_cell

        self.bottleneck_hidden = nn.Sequential(
            nn.Linear(seq_len*vocab_size, seq_len * vocab_size // 2),
            nn.ReLU(),
            nn.Linear(seq_len * vocab_size // 2, h_out * n_layers)
        )

        self.bottleneck_cell = nn.Sequential(
            nn.Linear(seq_len * vocab_size, seq_len * vocab_size // 2),
            nn.ReLU(),
            nn.Linear(seq_len * vocab_size // 2, h_cell * n_layers)
        )

    def forward(self, src):
        # src dimension (batch size, src len) - this is 2d tensor because it only has token indices.
        # embedded dimension (batch size, src len, emb dim)
        batch_size = src.shape[0]
        bert_out = self.bert(inputs_embeds=src)["last_hidden_state"]
        cell = self.bottleneck_cell(torch.flatten(bert_out, 1))
        hidden = self.bottleneck_hidden(torch.flatten(bert_out, 1))

        cell = cell.reshape(batch_size, self.n_layers, self.h_cell).permute(1, 0, 2).contiguous()
        hidden = hidden.reshape(batch_size, self.n_layers, self.h_out).permute(1, 0, 2).contiguous()

        # outputs dimension (batch size, src len, hid dim * n directions)
        # hidden dimension (n layers * n directions, batch size, hid dim)
        # cell dimension (n layers * n directions, batch size, hid dim)
        return hidden, cell


# Note that n directions in our model is 1.
# Here outputs is the hidden states for all the time steps in top layer of LSTM
# hidden and cell updates for each step and finally passed into Decoder.

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input dimension (batch size)
        input = input.unsqueeze(1)
        # input dimension (batch size, 1) - LSTM expects this shape
        embedded = self.dropout(self.embedding(input))
        # embedded dimension (batch size, 1, emd dim)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output dimension (batch size, 1, hid dim)
        # hidden, cell dimensions (n layers, batch size, hid dim)
        prediction = self.fc_out(output.squeeze(1))
        # output dimension (batch size, hid dim) - After Squeezing 1
        # prediction dimension (batch_size, output_dim)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # self.fc = nn.Linear(encoder.)

        # assert (encoder.hid_dim == decoder.hid_dim), "Hidden dimensions must match!"
        # assert (encoder.n_layers * 2 == decoder.n_layers), "Encoder and decoder must have the same number of layers!"

    def forward(self, src):
        batch_size = src.shape[0]
        trg_len = src.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        #input = trg[:, 0] # Passing the starting token (SOS)
        input = torch.zeros(batch_size, dtype=torch.long).to(self.device) # TODO: add SOS/PAD token
        for t in range(0, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output # Predictions upto target length
            top1 = output.argmax(1)
            # choose the max value from the logits and stored it in top1.
            input = top1
        return outputs



@numba.njit
def longest_consec(arr):
    n_rows, n_cols = arr.shape
    max_lengths = np.zeros(n_cols, dtype=np.int32)
    for col in range(n_cols):
        max_len = 0
        cur_len = 0
        for row in range(n_rows):
            if arr[row, col] == 1:
                cur_len += 1
                if cur_len > max_len:
                    max_len = cur_len
            else:
                cur_len = 0
        max_lengths[col] = max_len
    return max_lengths



class WindowIndexDataset(Dataset):
    def __init__(self, file_list, window_size=512, top_n=25, step_size=128, return_decode=False):
        self.entries = []
        self.window_size = window_size
        self.top_n = top_n
        self.step_size = step_size
        self.return_decode = return_decode
        for path in file_list:
            matrix = np.load(path, allow_pickle=True, mmap_mode='r')
            n_windows = (matrix.shape[0] - window_size) // step_size + 1
            self.entries.extend([(path, i) for i in range(n_windows)])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, window_idx = self.entries[idx]
        matrix = np.load(path, allow_pickle=True, mmap_mode='r')
        key = path.split("/")[2].split("_")[0]

        weights = np.load(f"training_data/weights/{key}_weights.npy", allow_pickle=True)

        start = window_idx * self.step_size
        end = start + self.window_size
        window_matrix_unmasked = matrix[start:end]

        consecutive_hit = longest_consec(window_matrix_unmasked)
        parent_support = window_matrix_unmasked.sum(axis=0)
        combined = consecutive_hit + parent_support
        top_parents = np.argpartition(combined, -self.top_n)[-self.top_n:]
        top_parents = top_parents[np.argsort(combined[top_parents])[::-1]]

        weights = np.array(weights, dtype=np.float16)
        weight_vector = weights[top_parents]
        weighted_window = window_matrix_unmasked[:, top_parents] * weight_vector
        #unweighted_window = window_matrix_unmasked[:, top_parents]

        if self.return_decode:
            decode_info = top_parents.tolist()
            return (
                torch.tensor(weighted_window, dtype=torch.float32),
                torch.tensor(decode_info, dtype=torch.int64)
            )
        else:
            return torch.tensor(weighted_window, dtype=torch.float32)

