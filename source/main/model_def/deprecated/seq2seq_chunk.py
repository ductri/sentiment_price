import torch
from torch import nn, optim

from model_def.deprecated.decoder import DecoderGreedyWithSrcInfer, AttnRawDecoderWithSrc
from model_def.encoder import Encoder, FlattenHiddenLSTM
from utils import pytorch_utils


class Seq2SeqChunk(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, start_idx, padding_idx, max_length):
        super(Seq2SeqChunk, self).__init__()

        pytorch_utils.register_buffer(self, 'lr_rate', 1e-3)
        pytorch_utils.register_buffer(self, 'max_length', max_length)
        pytorch_utils.register_buffer(self, 'chunk_size', 10)

        self.__start_idx_int = start_idx
        self.__padding_idx_int = padding_idx

        self.encoder = Encoder(vocab_size=src_vocab_size, is_bidirectional=False)
        _enc_output_size = 2*self.encoder.lstm_size.item() if self.encoder.is_bidirectional.item() else self.encoder.lstm_size.item()
        self.flatten_hidden_lstm = FlattenHiddenLSTM(lstm_num_layer=3, is_bidirectional=bool(self.encoder.is_bidirectional.item()))
        self.core_decoder = AttnRawDecoderWithSrc(vocab_size=tgt_vocab_size, enc_output_size=_enc_output_size,
                                                  enc_embedding_size=self.encoder.embedding_size.item())
        self.greedy_infer = DecoderGreedyWithSrcInfer(core_decoder=self.core_decoder)

        self.xent = None
        self.optimizer = None

        self.register_buffer('start_idx', torch.Tensor([start_idx]).long())
        self.register_buffer('padding_idx', torch.Tensor([[padding_idx]]).long())

    def chunk_forward(self, word_input, h_c, starts_idx, *args):
        """
        Encoding procedure is the same, but only decoding the first half of the sequence
        :param word_input: shape == (batch_size, max_len)
        :param h_c: tuple of (h, c). Set it None to indicate the start of the sequence
        :param starts_idx: Tensor shape == (batch)
        :param args:
        :return: Tensor shape == (batch, seq_len)
        """
        if h_c is not None:
            h_n, c_n, outputs = self.encoder(word_input)
        else:
            h_n, c_n, outputs = self.encoder(word_input, h_c)
        h_n, c_n = self.flatten_hidden_lstm(h_n, c_n)

        seq_len = word_input.size(1)
        assert seq_len % 2 == 0
        word_input = word_input[:, :int(seq_len/2)]
        enc_inputs = self.encoder.embedding(word_input)
        enc_inputs = enc_inputs.permute(1, 0, 2)
        output = self.greedy_infer(h_n, c_n, outputs, enc_inputs, starts_idx)
        return output, (h_n, c_n)

    def forward(self, word_input, *args):
        """

        :param word_input: shape == (batch_size, seq_len)
        :param args:
        :return: Tensor shape == (batch, seq_len)
        """
        __batch_size = word_input.size(0)

        input_chunks = self.__chunking_sequence(word_input)
        h_c = None
        output = []
        previous_starts_idx = self.start_idx.repeat(__batch_size)
        for i_chunk in input_chunks:
            output_chunk, h_c = self.chunk_forward(i_chunk, h_c, previous_starts_idx)
            output.append(output_chunk)
            previous_starts_idx = output_chunk[:, -1]

        output = torch.cat(output, dim=1)
        seq_len = word_input.size(1)
        output = output[:, :seq_len]
        return output

    def train(self, mode=True):
        if self.xent is None:
            self.xent = nn.CrossEntropyLoss(reduction='none')
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr_rate.item())
        super().train(mode)

    def get_loss_chunk(self, word_input, target, length, previous_starts_idx):
        """

        :param word_input: shape == (batch, seq_len)
        :param target: shape == (batch, seq_len/2)
        :param length: shape == (batch)
        :param previous_starts_idx: shape == (batch)
        :return: Tensor shape == (batch, seq_len/2)
        """
        assert target.size(1)*2 == word_input.size(1)
        __half_seq_len = target.size(1)
        __batch_size = word_input.size(0)

        enc_h_n, enc_c_n, enc_outputs = self.encoder(word_input)
        enc_h_n, enc_c_n = self.flatten_hidden_lstm(enc_h_n, enc_c_n)

        # shape == (batch_size, seq_len/2)
        dec_inputs = torch.cat((previous_starts_idx.view(__batch_size, 1), target[:, :-1]), dim=1)

        # shape == (seq_len/2, batch_size)
        dec_inputs = dec_inputs.permute(1, 0)

        enc_inputs = self.encoder.embedding(word_input[:, :__half_seq_len])
        # shape == (seq_len/2, batch, _)
        enc_inputs = enc_inputs.permute(1, 0, 2)

        # shape == (seq_len/2, batch_size, tgt_vocab_size)
        predict, _ = self.core_decoder(dec_inputs, (enc_h_n, enc_c_n), enc_outputs, enc_inputs, step=None)

        # shape == (batch_size, tgt_vocab_size, max_len+1)
        predict = predict.permute(1, 2, 0)

        dec_target = target
        loss = self.xent(predict, dec_target)
        __chunk_size = self.chunk_size.item()
        assert __chunk_size == word_input.size(1)

        loss_mask = pytorch_utils.length_to_mask(length, max_len=__half_seq_len, dtype=torch.float)
        loss = torch.mul(loss, loss_mask)

        return loss

    def get_loss(self, word_input, target, length):
        """

        :param word_input: shape == (batch, seq_len)
        :param target: shape == (batch, seq_len)
        :param length: shape == (batch)
        :return: Tensor shape == (batch, seq_len)
        """
        __max_length = self.max_length.item()
        __half_chunk_size = int(self.chunk_size.item() / 2)
        __batch_size = word_input.size(0)

        input_chunks = self.__chunking_sequence(word_input)
        target_chunks = self.__chunking_sequence(target)

        # shape == (batch, __max_length)
        mask = pytorch_utils.length_to_mask(length, max_len=__max_length)
        length_chunks = [torch.sum(mask[:, i:i+__half_chunk_size], dim=1)
                         for i in range(0, __max_length, __half_chunk_size)]
        loss = []
        previous_starts_idx = self.start_idx.repeat(__batch_size)
        for idx, (i_chunk, t_chunk) in enumerate(zip(input_chunks, target_chunks)):
            t_chunk = t_chunk[:, :__half_chunk_size]
            length_chunk = length_chunks[idx]
            loss.append(self.get_loss_chunk(i_chunk, t_chunk, length_chunk, previous_starts_idx))
            previous_starts_idx = t_chunk[:, -1]

        loss = torch.cat(loss, dim=1)
        loss = torch.div(loss.sum(dim=1), length.float())
        loss = loss.mean(dim=0)
        return loss

    def __chunking_sequence(self, word_input):
        """

        :param word_input:
        :return: List chunks
        """
        assert self.chunk_size % 2 == 0

        seq_len = word_input.size(1)
        batch_size = word_input.size(0)

        must_have = int(int(seq_len / (self.chunk_size / 2)) * (self.chunk_size / 2) + self.chunk_size)
        no_padding = must_have - seq_len
        padding = self.padding_idx.repeat(batch_size, no_padding)
        word_input = torch.cat((word_input, padding), dim=1)
        input_chunks = [word_input[:, i:i + self.chunk_size] for i in range(0, seq_len, int(self.chunk_size / 2))]
        return input_chunks

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
