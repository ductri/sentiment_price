import unittest

import numpy as np
import torch

from model_def.deprecated.seq2seq_decoder import RawDecoder, DecoderGreedyInfer, AttnRawDecoder, AttnRawDecoderWithSrc, \
    DecoderGreedyWithSrcInfer


class TestDecoder(unittest.TestCase):

    def test_raw_decoder(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10

        decoder = RawDecoder(vocab_size=vocab_size)

        inputs_idx = torch.randint(vocab_size, size=(max_length, batch_size))
        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)

        outputs, _ = decoder(inputs_idx, (h_n, c_n))

        self.assertEqual(outputs.size(), (max_length, batch_size, vocab_size))

    def test_raw_decoder_run_by_step(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10

        decoder = RawDecoder(vocab_size=vocab_size)
        decoder.eval()
        with torch.no_grad():
            inputs_idx = torch.randint(vocab_size, size=(max_length, batch_size))
            h_n = torch.rand(3, batch_size, 512)
            c_n = torch.rand(3, batch_size, 512)

            outputs1, (h1, c1) = decoder(inputs_idx, (h_n, c_n))

            h2, c2 = h_n, c_n
            outputs2 = []
            for step in range(inputs_idx.size(0)):
                inputs_idx_step = inputs_idx[step: step+1]
                output_, (h2, c2) = decoder(inputs_idx_step, (h2, c2))
                outputs2.append(output_)
            outputs2 = torch.cat(outputs2, dim=0)

            outputs1 = outputs1.numpy()
            outputs2 = outputs2.numpy()

            self.assertAlmostEqual(np.sum(np.abs((outputs1 - outputs2))), 0, places=5)
            self.assertAlmostEqual(np.sum(np.abs((h1.numpy() - h2.numpy()))), 0, places=5)
            self.assertAlmostEqual(np.sum(np.abs((c1.numpy() - c2.numpy()))), 0, places=5)

    def test_infer_decoder(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10

        raw_decoder = RawDecoder(vocab_size=vocab_size)
        decoder = DecoderGreedyInfer(core_decoder=raw_decoder, max_length=max_length, start_idx=0)

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)

        outputs = decoder(h_n, c_n)
        self.assertEqual(outputs.size(), (batch_size, max_length))

    def test_attn_decoder(self):
        vocab_size = 5
        batch_size = 2
        seq_len = 10
        enc_output_size = 13

        decoder = AttnRawDecoder(vocab_size=vocab_size, enc_output_size=enc_output_size)

        inputs_idx = torch.randint(vocab_size, size=(seq_len, batch_size))
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)

        outputs, _ = decoder(inputs_idx, (h_n, c_n), enc_outputs, None)

        self.assertEqual(outputs.size(), (seq_len, batch_size, vocab_size))

    def test_infer_attn_decoder(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10
        enc_output_size = 13
        seq_len = 3

        raw_decoder = AttnRawDecoder(vocab_size=vocab_size, enc_output_size=enc_output_size)
        decoder = DecoderGreedyInfer(core_decoder=raw_decoder, max_length=max_length, start_idx=0)

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)
        outputs = decoder(h_n, c_n, enc_outputs)
        self.assertEqual(outputs.size(), (batch_size, max_length))

    def test_attn_decoder_run_by_step(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10
        seq_len = 13
        enc_lstm_size = 512
        enc_output_size = enc_lstm_size*2

        decoder = AttnRawDecoder(vocab_size=vocab_size, enc_output_size=enc_output_size)
        decoder.eval()
        with torch.no_grad():
            inputs_idx = torch.randint(vocab_size, size=(max_length, batch_size))
            h_n = torch.rand(3, batch_size, enc_lstm_size)
            c_n = torch.rand(3, batch_size, enc_lstm_size)
            enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)

            outputs1, (h1, c1) = decoder(inputs_idx, (h_n, c_n), enc_outputs, None)

            h2, c2 = h_n, c_n
            outputs2 = []
            for step in range(inputs_idx.size(0)):
                inputs_idx_step = inputs_idx[step: step+1]
                output_, (h2, c2) = decoder(inputs_idx_step, (h2, c2), enc_outputs, step)
                outputs2.append(output_)
            outputs2 = torch.cat(outputs2, dim=0)

            outputs1 = outputs1.numpy()
            outputs2 = outputs2.numpy()

            self.assertAlmostEqual(np.sum(np.abs((outputs1 - outputs2))), 0, places=5)
            self.assertAlmostEqual(np.sum(np.abs((h1.numpy() - h2.numpy()))), 0, places=5)
            self.assertAlmostEqual(np.sum(np.abs((c1.numpy() - c2.numpy()))), 0, places=5)

    def test_attn_with_src_decoder(self):
        vocab_size = 5
        batch_size = 2
        seq_len = 10
        enc_output_size = 13
        enc_embedding_size = 11

        decoder = AttnRawDecoderWithSrc(vocab_size=vocab_size, enc_output_size=enc_output_size,
                                        enc_embedding_size=enc_embedding_size)

        inputs_idx = torch.randint(vocab_size, size=(seq_len, batch_size))
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)
        enc_inputs = torch.randn(seq_len, batch_size, enc_embedding_size)

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)

        outputs, _ = decoder(inputs_idx, (h_n, c_n), enc_outputs, enc_inputs, None)

        self.assertEqual(outputs.size(), (seq_len, batch_size, vocab_size))

    def test_infer_attn_with_src_decoder(self):
        vocab_size = 5
        batch_size = 2
        enc_output_size = 13
        seq_len = 3
        enc_embedding_size = 11

        raw_decoder = AttnRawDecoderWithSrc(vocab_size=vocab_size, enc_output_size=enc_output_size,
                                            enc_embedding_size=enc_embedding_size)
        decoder = DecoderGreedyWithSrcInfer(core_decoder=raw_decoder, start_idx=0)

        h_n = torch.rand(3, batch_size, 512)
        c_n = torch.rand(3, batch_size, 512)
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)
        enc_inputs = torch.randn(seq_len, batch_size, enc_embedding_size)
        outputs = decoder(h_n, c_n, enc_outputs, enc_inputs)
        self.assertEqual(outputs.size(), (batch_size, seq_len))

    def test_attn_with_src_decoder_run_by_step(self):
        vocab_size = 5
        batch_size = 2
        max_length = 10
        seq_len = 13
        enc_lstm_size = 512
        enc_output_size = enc_lstm_size*2
        enc_embedding_size = 11

        core_decoder = AttnRawDecoderWithSrc(vocab_size=vocab_size, enc_output_size=enc_output_size,
                                        enc_embedding_size=enc_embedding_size)

        core_decoder.eval()
        with torch.no_grad():
            inputs_idx = torch.randint(vocab_size, size=(max_length, batch_size))
            h_n = torch.rand(3, batch_size, enc_lstm_size)
            c_n = torch.rand(3, batch_size, enc_lstm_size)
            enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)
            enc_inputs = torch.randn(seq_len, batch_size, enc_embedding_size)

            outputs1, (h1, c1) = core_decoder(inputs_idx, (h_n, c_n), enc_outputs, enc_inputs, None)

            h2, c2 = h_n, c_n
            outputs2 = []
            for step in range(inputs_idx.size(0)):
                inputs_idx_step = inputs_idx[step: step+1]
                output_, (h2, c2) = core_decoder(inputs_idx_step, (h2, c2), enc_outputs, enc_inputs[step:step+1], step)
                outputs2.append(output_)
            outputs2 = torch.cat(outputs2, dim=0)

            outputs1 = outputs1.numpy()
            outputs2 = outputs2.numpy()

            self.assertAlmostEqual(np.sum(np.abs((outputs1 - outputs2))), 0, places=5)
            self.assertAlmostEqual(np.sum(np.abs((h1.numpy() - h2.numpy()))), 0, places=5)
            self.assertAlmostEqual(np.sum(np.abs((c1.numpy() - c2.numpy()))), 0, places=5)


if __name__ == '__main__':
    unittest.main()


