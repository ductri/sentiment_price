import torch
from torch import nn, optim

from model_def.deprecated.seq2seq_decoder import DecoderGreedyInfer, RawDecoder
from model_def.deprecated.seq2seq_encoder import Encoder, FlattenHiddenLSTM
from utils import pytorch_utils


class Seq2Seq(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, start_idx, end_idx):
        super(Seq2Seq, self).__init__()
        self.lr_rate = 1e-3
        self.max_length = 100
        self.__start_idx_int = start_idx

        self.encoder = Encoder(vocab_size=src_vocab_size)
        self.flatten_hidden_lstm = FlattenHiddenLSTM(lstm_num_layer=3, is_bidirectional=self.encoder.is_bidirectional)
        self.core_decoder = RawDecoder(vocab_size=tgt_vocab_size)
        self.greedy_infer = DecoderGreedyInfer(core_decoder=self.core_decoder, max_length=self.max_length,
                                               start_idx=start_idx)

        self.xent = None
        self.optimizer = None

        self.register_buffer('start_idx', torch.Tensor([[start_idx]]).long())
        self.register_buffer('end_idx', torch.Tensor([[end_idx]]).long())

    def forward(self, word_input, *input):
        """

        :param word_input: shape == (batch_size, max_len)
        :param start_idx: int scala
        :param end_idx: int scala
        :param input:
        :return:
        """
        h_n, c_n, _ = self.encoder(word_input)
        h_n, c_n = self.flatten_hidden_lstm(h_n, c_n)
        output = self.greedy_infer(h_n, c_n, self.__start_idx_int)
        return output

    def train(self, mode=True):
        if self.xent is None:
            self.xent = nn.CrossEntropyLoss(reduction='none')
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr_rate)
        super().train(mode)

    def get_loss(self, word_input, target, length):
        """

        :param word_input: shape == (batch_size, max_len)
        :param target: shape == (batch_size, max_len)
        :param length: shape == (batch_size)
        :return:
        """
        enc_h_n, enc_c_n, _ = self.encoder(word_input)
        enc_h_n, enc_c_n = self.flatten_hidden_lstm(enc_h_n, enc_c_n)
        batch_size = enc_h_n.size(1)

        init_words = self.start_idx.repeat(batch_size, 1)
        # shape == (batch_size, max_len + 1)
        dec_input = torch.cat((init_words, target), dim=1)

        # shape == (max_len + 1, batch_size)
        dec_input = dec_input.permute(1, 0)

        # shape == (max_len+1, batch_size, tgt_vocab_size)
        predict, _ = self.core_decoder(dec_input, (enc_h_n, enc_c_n))

        # shape == (batch_size, tgt_vocab_size, max_len+1)
        predict = predict.permute(1, 2, 0)

        end_words = self.end_idx.repeat(batch_size, 1)
        dec_target = torch.cat((target, end_words), dim=1)

        loss = self.xent(predict, dec_target)
        loss_mask = pytorch_utils.length_to_mask(length+1, max_len=self.max_length+1, dtype=torch.float)
        loss = torch.mul(loss, loss_mask)
        loss = torch.div(loss.sum(dim=1), (length+1).float())
        loss = loss.mean(dim=0)
        return loss

    def train_batch(self, word_input, target, length):
        """

        :param word_input: shape == (batch_size, max_len)
        :param target: shape == (batch_size, max_len)
        :return:
        """
        self.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(word_input, target, length)
        loss.backward()
        self.optimizer.step()

        return loss.item()
