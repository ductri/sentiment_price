from torch import nn

from utils import pytorch_utils


class Encoder(nn.Module):

    def __init__(self, vocab_size, is_bidirectional=True):
        """

        :param vocab_size:
        :param is_bidirectional:
        """
        super(Encoder, self).__init__()

        self.embedding_size = 512
        self.lstm_size = 512
        self.lstm_num_layer = 3
        self.is_bidirectional = is_bidirectional
        pytorch_utils.register_buffer(self, 'dropout_rate', 0.3)
        __dropout_rate = self.dropout_rate.item()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.lstm_size, num_layers=self.lstm_num_layer,
                            bidirectional=self.is_bidirectional, dropout=__dropout_rate)

        self.dropout = nn.Dropout(__dropout_rate)

    def forward(self, word_input, *args):
        """

        :param word_input: shape == (batch_size, max_seq_len)
        :return: h_n, c_n, each has shape == (num_layers * num_directions, batch, hidden_size)
        """
        embedding = self.embedding(word_input)
        embedding = self.dropout(embedding)

        # shape == (max_word_len, batch_size, hidden_size)
        word_embed_permuted = embedding.permute(1, 0, 2)

        output, (h_n, c_n) = self.lstm(word_embed_permuted, *args)

        return output, (h_n, c_n)


class FlattenHiddenLSTM(nn.Module):
    def __init__(self, lstm_num_layer, is_bidirectional):
        """
        Flatten last hidden from bidirectional LSTM to feed into unidirectional LSTM .
        :param lstm_num_layer:
        """
        super(FlattenHiddenLSTM, self).__init__()
        self.lstm_num_layer = lstm_num_layer
        self.is_bidirectional = is_bidirectional

    def forward(self, h_n, c_n):
        """
        Input: last hidden from LSTM
        :param h_n:
        :param c_n:
        :return: shape == (batch_size, _)
        """
        h_n, c_n = self.__flatten(h_n), self.__flatten(c_n)
        return h_n, c_n

    def __flatten(self, x):
        """

        :param x: shape == (_, batch, lstm_size)
        :return: shape == (batch, _)
        """
        batch_size = x.size(1)
        # shape == (batch, num_lyaer)
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(batch_size, -1)
        return x
