import torch
from torch import nn, optim

from model_def.encoder import Encoder, FlattenHiddenLSTM


class LSTMBaseline(nn.Module):

    def __init__(self, vocab_size, no_class):
        super(LSTMBaseline, self).__init__()
        self.encoder = Encoder(vocab_size, is_bidirectional=True)
        self.flatten_hidden_lstm = FlattenHiddenLSTM(self.encoder.lstm_num_layer, self.encoder.is_bidirectional)
        self.fc = nn.Linear(in_features=self.encoder.lstm_size*2*self.encoder.lstm_num_layer*2, out_features=512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output_mapping = nn.Linear(in_features=512, out_features=no_class)
        self.softmax = nn.Softmax(dim=1)

        self.xent = None
        self.optimizer = None

    def _inner_forward(self, word_input, *args):
        """

        :param word_input: shape == (batch, seq_len)
        :param args:
        :return: logits Tensor shape == (batch, no_class)
        """
        _, h_c = self.encoder(word_input)
        h_c = self.flatten_hidden_lstm(h_c[0], h_c[1])
        inner = torch.cat(h_c, dim=1)
        inner = self.dropout(inner)
        inner = self.fc(inner)
        inner = self.relu(inner)
        output = self.output_mapping(inner)
        return output

    def forward(self, word_input, *args):
        logits = self._inner_forward(word_input)
        return self.softmax(logits)

    def train(self, mode=True):
        if self.xent is None:
            self.xent = nn.CrossEntropyLoss(reduction='none')
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        super().train(mode)

    def get_loss(self, word_input, target):
        """

        :param word_input: shape == (batch, seq_len)
        :param target: shape == (batch)
        :return:
        """
        logits = self._inner_forward(word_input)
        loss = self.xent(logits, target)
        loss = torch.mean(loss, dim=0)
        return loss

    def train_batch(self, word_input, target):
        self.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(word_input, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
