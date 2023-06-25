import torch
import torch.nn as nn


class EmformerHead(nn.Module):
    def __init__(self, emformer, hidden_size, num_layers, class_num=2, dropout=0.1):
        self.emformer = emformer
        self.ff_dropout = dropout
        self.window = list(emformer.children())[0].kernel_size[0]
        self.emformer_size = list(emformer.children())[-1][-1].pos_ff[-2].out_features
        self.head = nn.Sequential(
            self._make_layer(self.emformer_size, hidden_size),
            *[
                self._make_layer(hidden_size, hidden_size)
                for _ in range(num_layers - 2)
            ],
            nn.Linea(hidden_size, class_num)
        )

    def _make_layer(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.ff_dropout),
        )

    def forward(self, inputs, lengths):
        emb, lengths = self.emformer(inputs, lengths)
        out = self.head(emb)
        return out, emb, lengths

    def infer(self, inputs, lengths, states=None):
        emb, lengths, states = self.emformer(inputs, lengths, states)
        out = self.head(emb)
        return out, emb, lengths, states


class RawHead(nn.Module):
    def __init__(self, hidden_size, num_layers, class_num=1, dropout=0.01):
        super().__init__()
        self.ff_dropout = dropout
        self.hidden_size = hidden_size
        self.feat = nn.Sequential(
            *[
                self._make_layer(hidden_size, hidden_size)
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_size, hidden_size)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.ff_dropout),
            nn.Linear(hidden_size, class_num),
            nn.Sigmoid()
        )

    def _make_layer(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.ff_dropout),
        )

    def forward(self, inputs):
        out = self.feat(inputs)
        conf = self.head(out)
        return out, conf
