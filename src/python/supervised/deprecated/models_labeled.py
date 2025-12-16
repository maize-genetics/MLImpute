# This file stores all the model architectures used
import torch
from transformers import BertConfig, BertLMHeadModel, GenerationMixin
from transformers import PreTrainedModel
from torch import nn

class DecoderOnlyConfig(BertConfig):
    """Config class for imputation decoders. Same as BertConfig, but with the additional parameter num_parents."""
    def __init__(self, num_parents=24, **kwargs):
        self.num_parents = num_parents
        super().__init__(**kwargs)


# NOTE: when using BERT as a decoder with GenerationMixin, DO NOT use caches. It will crash.
class DecoderOnlyModel(PreTrainedModel, GenerationMixin):
    """BertLMHeadModel with a linear projection layer at the start to allow an arbitrary hidden size."""
    config_class = DecoderOnlyConfig

    def __init__(self, config):
        super().__init__(config)
        self.proj = torch.nn.Linear(config.num_parents, config.hidden_size)
        self.model = BertLMHeadModel(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
            **loss_kwargs,
    ):
        if encoder_hidden_states is not None:
            projection = self.proj(encoder_hidden_states)
        else:
            projection = encoder_hidden_states

        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                          position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                          encoder_hidden_states=projection, encoder_attention_mask=encoder_attention_mask,
                          labels=labels, past_key_values=past_key_values, use_cache=use_cache,
                          output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                          return_dict=return_dict, cache_position=cache_position, **loss_kwargs)


class LstmSegmentationEncDec(nn.Module):
    """Encoder-decoder model where both halves are LSTMs"""
    def __init__(self, num_parents, hid_dim, n_layers, dropout):
        """Parameters:
            num_parents -- number of parents in the PS4G file
            hid_dim -- hidden dimesion for the LSTMs
            n_layers -- number of layers in each LSTM
            dropout -- dropout rate
        """

        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.enc = nn.LSTM(num_parents, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dec = nn.LSTM(1, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, 1)
        self.forward_func = self.train_forward

    def train_mode(self):
        """Set model to use teacher-forcing for training"""
        self.forward_func = self.train_forward

    def eval_mode(self):
        """Set model for inference: stop only when stopping criteria are met"""
        self.forward_func = self.inference_forward

    # labels shape: (N, L, 1)
    def train_forward(self, src, labels):
        """Forward function for training (with teacher forcing).
            Parameters:
                src: input matrix, shape (batch-size, context-window-size, num_parents)
                labels: matrix of crossover point indices, shape (batch-size, L, 1)
        """
        batch_size = src.shape[0]
        label_length = labels.shape[1]
        device = src.device

        embedded = self.dropout(src)
        _, (hidden, cell) = self.enc(embedded)

        outputs = torch.zeros(batch_size, label_length, 1).to(device)
        input = torch.zeros((batch_size, 1, 1)).to(device)

        for t in range(0, label_length):
            dec_out, (hidden, cell) = self.dec(input, (hidden, cell))
            output = self.fc_out(dec_out)

            outputs[:, t, :] = output.squeeze(1) # Predictions upto target length
            input = labels[:, t].unsqueeze(1).unsqueeze(2) # teacher forcing

        return outputs

    def inference_forward(self, src, labels=None):
        """Forward function for inference, with no label input.
            Model stops predicting crossovers when all members of batch reach the stopping criteria -
            a value past the end of the context window is predicted, or the maximum length is reached.
            Maximum length is the length of the context window (i.e., each bin contains a crossover point)
            Parameters:
                src: input matrix, shape (batch-size, context-window-size, num_parents)
                labels: unused, here to match parameters with train_forward()
        """
        batch_size = src.shape[0]
        label_length = src.shape[1]
        device = src.device

        embedded = self.dropout(src)
        _, (hidden, cell) = self.enc(embedded)

        outputs = torch.zeros(batch_size, label_length, 1).to(device)
        input = torch.zeros((batch_size, 1, 1)).to(device)

        stop_criteria_unmet = torch.ones((batch_size, 1)).to(device)

        for t in range(0, label_length):
            dec_out, (hidden, cell) = self.dec(input, (hidden, cell))
            output = self.fc_out(dec_out)

            stop_criteria_unmet = stop_criteria_unmet * (output.squeeze(1) <= label_length)

            outputs[:, t, :] = output.squeeze(1) * stop_criteria_unmet   # Predictions upto target length

            if stop_criteria_unmet.sum() == 0:
                break

            input = output

        return outputs[:, 0:t+1, :]


    def forward(self, src, labels=None):
        return self.forward_func(src, labels)


class CNN(nn.Module):
    """Basic CNN for binary categorical prediction"""
    def __init__(self, input_dim=24, output_dim=1, sigmoid=True):
        super().__init__()
        self.sigmoid = sigmoid
        self.num_parents = input_dim

        if sigmoid:
            self.net = nn.Sequential(
                nn.Conv1d(self.num_parents, 128, 8, 1),
                nn.ReLU(),
                nn.MaxPool1d(2, 1),
                nn.Conv1d(128, 512, 4, 1),
                nn.ReLU(),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(512, 1024, 4, 1),
                nn.ReLU(),
                nn.MaxPool1d(2, 2),
                nn.Flatten(),
                nn.Linear(1024 * 3, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
                nn.Sigmoid()
            )
        else:
            # This CNN has three convolutional layers and 3 fully connected layers
            self.net = nn.Sequential(
                nn.Conv1d(self.num_parents, 128, 8, 1),
                nn.ReLU(),
                nn.MaxPool1d(2, 1),
                nn.Conv1d(128, 512, 4, 1),
                nn.ReLU(),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(512, 1024, 4, 1),
                nn.ReLU(),
                nn.MaxPool1d(2, 2),
                nn.Flatten(),
                nn.Linear(1024 * 3, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )

    def forward(self, src):
        self.net(src.permute(0, 2, 1))


class CNN2D(nn.Module):
    """Basic CNN for binary categorical prediction"""
    def __init__(self, output_dim=1, sigmoid=True):
        super().__init__()
        self.sigmoid = sigmoid

        if sigmoid:
            self.net = nn.Sequential(
                nn.Conv2d(1, 64, (8, 8), 1),
                nn.ReLU(),
                nn.MaxPool1d((2, 2), 1),
                nn.Conv2d(64, 256, (4, 4), 1),
                nn.ReLU(),
                nn.MaxPool1d((2, 2), 1),
                nn.Conv2d(256, 1024, (4, 12), 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1024 * 17 * 1, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
                nn.Sigmoid()
            )
        else:
            # This CNN has three convolutional layers and 3 fully connected layers
            self.net = nn.Sequential(
                nn.Conv2d(1, 64, (8, 8), 1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), 1),
                nn.Conv2d(64, 256, (4, 4), 1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), 1),
                nn.Conv2d(256, 1024, (4, 12), 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1024 * 17 * 1, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )

    def forward(self, src):
        return self.net(src.unsqueeze(1))


class FullyConnected(nn.Module):
    """Basic fully conencted network"""
    def __init__(self, input_dim=768, output_dim=1, hidden_dim=256, num_layers=5, sigmoid=True):
        super().__init__()
        self.sigmoid = sigmoid

        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim))

        for idx in range(num_layers-1):
            self.net.append(nn.ReLU())
            if idx == num_layers-2:
                self.net.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.net.append(nn.Linear(hidden_dim, hidden_dim))

        if sigmoid:
            self.net.append(nn.Sigmoid())

    def forward(self, src):
        return self.net(src.flatten(1))

