from torch import nn

from utils import pytorch_utils


class Encoder(nn.Module):

    def __init__(self, vocab_size, is_bidirectional=True):
        """
        Output last hidden state of LSTM
        :param vocab_size:
        """
        super(Encoder, self).__init__()

        pytorch_utils.register_buffer(self, 'embedding_size', 256)
        pytorch_utils.register_buffer(self, 'lstm_size', 512)
        pytorch_utils.register_buffer(self, 'lstm_num_layer', 3)
        pytorch_utils.register_buffer(self, 'is_bidirectional', is_bidirectional)
        pytorch_utils.register_buffer(self, 'dropout_rate', 0.3)

        __embedding_size = self.embedding_size.item()
        __lstm_size = self.lstm_size.item()
        __lstm_num_layer = self.lstm_num_layer.item()
        __is_bidirectional = bool(self.is_bidirectional.item())
        __dropout_rate = self.dropout_rate.item()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=__embedding_size)
        self.lstm = nn.LSTM(input_size=__embedding_size, hidden_size=__lstm_size, num_layers=__lstm_num_layer,
                            bidirectional=__is_bidirectional, dropout=__dropout_rate)

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

        return h_n, c_n, output


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
        :return: shape == (batch_size, hidden_size)
        """
        h_n, c_n = self.__flatten(h_n), self.__flatten(c_n)
        return h_n, c_n

    def __flatten(self, x):
        batch_size = x.size(1)
        hidden_size = x.size(2)
        num_direction = 2 if self.is_bidirectional else 1
        x = x.view(self.lstm_num_layer, num_direction, batch_size, hidden_size)
        x = x.mean(dim=1)
        return x
