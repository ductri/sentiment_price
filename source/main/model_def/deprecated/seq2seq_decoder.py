import torch
from torch import nn

from model_def.attention import Attention


class DecoderGreedyInfer(nn.Module):

    def __init__(self, core_decoder, max_length, start_idx):
        """
        Output a fixed vector with size of `output_size` for each doc
        :param vocab_size:
        :param max_length: scala int
        :param start_idx: scala int
        """
        super(DecoderGreedyInfer, self).__init__()
        self.core_decoder = core_decoder
        self.register_buffer('start_idx', torch.Tensor([[start_idx]]))
        self.register_buffer('max_length', max_length)

    def forward(self, enc_h_n, enc_c_n, *args):
        """

        :param enc_h_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param enc_c_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param args:
        :return:
        """
        with torch.no_grad():
            batch_size = enc_c_n.size(1)
            decoder_output = torch.zeros(batch_size, self.max_length)

            current_word = self.start_idx.repeat(1, batch_size).long()
            h_n, c_n = (enc_h_n, enc_c_n)
            for step in range(self.max_length):
                # shape == (1, batch_size, vocab_size)
                args = args + (step,)
                output, (h_n, c_n) = self.core_decoder(current_word, (h_n, c_n), *args)

                # shape == (1, batch_size)
                current_word = torch.argmax(output, dim=2)

                decoder_output[:, step] = current_word[0]

            return decoder_output.int()


class DecoderGreedyWithSrcInfer(nn.Module):

    def __init__(self, core_decoder, start_idx):
        """
        Output a fixed vector with size of `output_size` for each doc
        :param vocab_size:
        :param max_length: scala int
        :param start_idx: scala int
        """
        super(DecoderGreedyWithSrcInfer, self).__init__()
        self.core_decoder = core_decoder
        self.register_buffer('start_idx', torch.Tensor([[start_idx]]))

    def forward(self, enc_h_n, enc_c_n, enc_outputs, enc_inputs):
        """

        :param enc_h_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param enc_c_n: shape = (num_direction*num_layers, batch_size, hidden_size)
        :param enc_outputs: shape = (seq_len, batch, _)
        :param enc_inputs: shape = (seq_len, batch, enc_embedding_size)
        :param args:
        :return: shape == (batch, seq_len)
        """
        with torch.no_grad():
            batch_size = enc_c_n.size(1)
            seq_len = enc_inputs.size(0)
            decoder_output = torch.zeros(batch_size, seq_len)

            current_word = self.start_idx.repeat(1, batch_size).long()
            h_n, c_n = (enc_h_n, enc_c_n)

            for step in range(seq_len):
                # shape == (1, batch_size, vocab_size)
                output, (h_n, c_n) = self.core_decoder(current_word, (h_n, c_n), enc_outputs,
                                                       enc_inputs[step: step + 1], step)

                # shape == (1, batch_size)
                current_word = torch.argmax(output, dim=2)

                decoder_output[:, step] = current_word[0]

            return decoder_output.int()


class RawDecoder(nn.Module):
    def __init__(self, vocab_size):
        """
        Common use for both Training and Inference
        :param vocab_size:
        """
        super(RawDecoder, self).__init__()
        self.register_buffer('embedding_size', 256)
        self.register_buffer('lstm_size', 512)
        self.register_buffer('lstm_num_layer', 3)
        self.register_buffer('dropout_rate', 0.3)

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.lstm_size, num_layers=self.lstm_num_layer,
                            bidirectional=False, dropout=self.dropout_rate)
        self.output_mapping = nn.Linear(self.lstm_size, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs_idx, h_n_c_n, *args):
        """
        Can input in 2 flavors: step by step, or whole sequence
        :param inputs_idx: shape == (max_length, batch_size)
        :param h_n_c_n: tuple of (h_n, c_n) from LSTM. Each has size of (num_layers * num_directions, batch, hidden_size)
        :param args:
        :return: output shape == (max_length, batch, vocab_size)
        """

        # shape == (max_length, batch_size, hidden_size)
        embedding_input = self.embedding(inputs_idx)
        # output shape == (max_length, batch, num_directions * hidden_size)
        outputs, (h_n, c_n) = self.lstm(embedding_input, h_n_c_n)
        outputs = self.dropout(outputs)
        outputs = self.output_mapping(outputs)
        return outputs, (h_n, c_n)


class AttnRawDecoder(nn.Module):
    def __init__(self, vocab_size, enc_output_size):
        """
        TODO WARNING NotImplement input feeding
        Common use for both Training and Inference
        :param vocab_size:
        """
        super(AttnRawDecoder, self).__init__()
        self.register_buffer('embedding_size', 256)
        self.register_buffer('lstm_size', 512)
        self.register_buffer('lstm_num_layer', 3)
        self.register_buffer('dropout_rate', 0.3)
        self.register_buffer('half_window_size', 50)

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.lstm_size, num_layers=self.lstm_num_layer,
                            bidirectional=False, dropout=self.dropout_rate)
        self.attention = Attention(enc_output_size=enc_output_size, dec_output_size=self.lstm_size)
        self.output_mapping = nn.Linear(self.lstm_size+enc_output_size, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs_idx, h_n_c_n, enc_outputs, step, *args):
        """
        Implemented by running step by step
        :param inputs_idx: shape == (seq_len, batch_size)
        :param h_n_c_n: tuple of (h_n, c_n) from LSTM. Each has size of (num_layers * num_directions, batch, hidden_size)
        :param enc_outputs: shape == (seq_len, batch, hidden_size)
        :param step:
        :param args:
        :return: output shape == (seq_len, batch, vocab_size)
        """
        # shape == (seq_len, batch_size, hidden_size)
        embedding_input = self.embedding(inputs_idx)

        if step is None:
            outputs = []
            for step in range(inputs_idx.size(0)):
                inputs_idx_step = embedding_input[step: step + 1]
                output_, h_n_c_n = self.lstm(inputs_idx_step, h_n_c_n)
                output_ = output_[0]
                local_enc_outputs = enc_outputs[max(0, step - self.half_window_size):step + self.half_window_size]
                output_, _ = self.attention(local_enc_outputs, output_)
                output_ = output_.view(1, *output_.size())
                outputs.append(output_)
            # output shape == (seq_len, batch, size)
            outputs = torch.cat(tuple(outputs), dim=0)
        else:
            inputs_idx_step = embedding_input
            output_, h_n_c_n = self.lstm(inputs_idx_step, h_n_c_n)
            output_ = output_[0]
            local_enc_outputs = enc_outputs[max(0, step - self.half_window_size):step + self.half_window_size]
            output_, _ = self.attention(local_enc_outputs, output_)
            outputs = output_.view(1, *output_.size())
        outputs = self.dropout(outputs)
        outputs = self.output_mapping(outputs)
        return outputs, h_n_c_n


class AttnRawDecoderWithSrc(nn.Module):
    def __init__(self, vocab_size, enc_output_size, enc_embedding_size):
        """
        Common use for both Training and Inference
        :param vocab_size:
        """
        super(AttnRawDecoderWithSrc, self).__init__()
        self.register_buffer('embedding_size', 256)
        self.register_buffer('lstm_size', 512)
        self.register_buffer('lstm_num_layer', 3)
        self.register_buffer('dropout_rate', 0.3)
        self.register_buffer('half_window_size', 3)

        self.dec_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size+enc_embedding_size, hidden_size=self.lstm_size, num_layers=self.lstm_num_layer,
                            bidirectional=False, dropout=self.dropout_rate)
        self.attention = Attention(enc_output_size=enc_output_size, dec_output_size=self.lstm_size)
        _shrink_size = 512
        self.shrink_mapping = nn.Linear(self.lstm_size + enc_output_size, _shrink_size)
        self.output_mapping = nn.Linear(_shrink_size, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs_idx, h_n_c_n, enc_outputs, enc_inputs, step, *args):
        """
        Implemented by running step by step
        :param inputs_idx: shape == (seq_len, batch_size)
        :param h_n_c_n: tuple of (h_n, c_n) from LSTM. Each has size of (num_layers * num_directions, batch, hidden_size)
        :param enc_outputs: shape == (seq_len, batch, hidden_size)
        :param enc_inputs: shape == (seq_len, batch_size, encoder_embedding_size). In case of step != None, seq_len==1
        :param step:
        :param args:
        :return: output shape == (seq_len, batch, vocab_size)
        """
        # shape == (seq_len, batch, decoder_embedding_size)
        embedding_input = self.dec_embedding(inputs_idx)

        if step is None:
            # This case for training
            outputs = []
            for step in range(inputs_idx.size(0)):
                # shape == (1, batch_size, _)
                inputs_idx_step = torch.cat((embedding_input[step: step + 1], enc_inputs[step: step + 1]), dim=2)

                output_, h_n_c_n = self.lstm(inputs_idx_step, h_n_c_n)
                output_ = output_[0]
                local_enc_outputs = enc_outputs[max(0, step - self.half_window_size):step + self.half_window_size]
                output_, _ = self.attention(local_enc_outputs, output_)

                # shape == (1, batch_size, _)
                output_ = output_.view(1, *output_.size())

                outputs.append(output_)
            # output shape == (seq_len, batch, _)
            outputs = torch.cat(tuple(outputs), dim=0)
        else:
            # This case for inference because we haven't known step in advance
            assert enc_inputs.size(0) == 1
            assert embedding_input.size(0) == 1

            inputs_idx_step = torch.cat((embedding_input, enc_inputs), dim=2)
            output_, h_n_c_n = self.lstm(inputs_idx_step, h_n_c_n)
            output_ = output_[0]
            local_enc_outputs = enc_outputs[max(0, step - self.half_window_size):step + self.half_window_size]
            output_, _ = self.attention(local_enc_outputs, output_)

            # shape == (1, batch_size, _)
            output_ = output_.view(1, *output_.size())

            outputs = output_

        outputs = self.shrink_mapping(outputs)
        outputs = self.dropout(outputs)
        outputs = self.output_mapping(outputs)
        return outputs, h_n_c_n

