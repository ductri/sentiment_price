import torch
from torch import nn, optim

from model_def.encoder import Encoder, FlattenHiddenLSTM
from model_def.attention import Attention


class LSTMAttention(nn.Module):

    def __init__(self, vocab_size, no_class):
        super(LSTMAttention, self).__init__()
        self.encoder = Encoder(vocab_size, is_bidirectional=True)
        __temp_1 = 512
        self.attention = Attention(enc_output_size=self.encoder.lstm_size*2, dec_output_size=__temp_1)

        self.flatten_hidden_lstm = FlattenHiddenLSTM(self.encoder.lstm_num_layer, self.encoder.is_bidirectional)
        self.fc1 = nn.Linear(in_features=self.encoder.lstm_size * 2 * self.encoder.lstm_num_layer * 2, out_features=__temp_1)
        self.fc2 = nn.Linear(in_features=self.encoder.lstm_size*2+__temp_1, out_features=__temp_1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output_mapping = nn.Linear(in_features=__temp_1, out_features=no_class)
        self.softmax = nn.Softmax(dim=1)

        self.xent = None
        self.optimizer = None

    def __inner_forward(self, word_input, *args):
        """

        :param word_input: shape == (batch, seq_len)
        :param args:
        :return: logits Tensor shape == (batch, no_class)
        """
        enc_outputs, h_c = self.encoder(word_input)
        h_c = self.flatten_hidden_lstm(h_c[0], h_c[1])
        inner = torch.cat(h_c, dim=1)
        inner = self.dropout(inner)
        inner = self.fc1(inner)
        inner, _ = self.attention(enc_outputs, inner)
        inner = self.fc2(inner)
        inner = self.relu(inner)
        output = self.output_mapping(inner)
        return output

    def forward(self, word_input, *args):
        logits = self.__inner_forward(word_input)
        return self.softmax(logits)

    def train(self, mode=True):
        if self.xent is None:
            self.xent = nn.CrossEntropyLoss(reduction='mean', weight=torch.tensor([1., 3.]).cuda())
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        super().train(mode)

    def get_loss(self, word_input, target):
        """

        :param word_input: shape == (batch, seq_len)
        :param target: shape == (batch)
        :return:
        """
        logits = self.__inner_forward(word_input)
        loss = self.xent(logits, target)
        return loss

    def train_batch(self, word_input, target):
        self.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(word_input, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
