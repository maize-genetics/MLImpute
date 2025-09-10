# modified from https://medium.com/@challamalla5sunil/building-a-machine-translation-system-with-encoder-decoder-architecture-in-pytorch-b7373fee1760

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import random
import math
import numpy as np
import time

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, src):
        # src dimension (batch size, src len) - this is 2d tensor because it only has token indices.
        embedded = self.dropout(self.embedding(src))
        # embedded dimension (batch size, src len, emb dim)
        batch_size = src.shape[0]
        hidden = torch.zeros(self.n_layers, batch_size, self.hid_dim).to(self.device)
        cell = torch.zeros(self.n_layers, batch_size, self.hid_dim).to(self.device)
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))

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
        assert (encoder.hid_dim == 2 * decoder.hid_dim), "Hidden dimensions must match!"
        assert (encoder.n_layers == decoder.n_layers), "Encoder and decoder must have the same number of layers!"

    def forward(self, src):
        batch_size = src.shape[0]
        trg_len = src.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        #input = trg[:, 0] # Passing the starting token (SOS)
        input = torch.LongTensor([0]).to(self.device) # TODO: add SOS/PAD token
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output # Predictions upto target length
            top1 = output.argmax(1)
            # choose the max value from the logits and stored it in top1.
            input = top1
        return outputs
