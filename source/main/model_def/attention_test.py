import unittest
import torch

from model_def.attention import Attention


class TestAttention(unittest.TestCase):

    def test_forward(self):
        enc_output_size = 10
        dec_output_size = 15
        batch_size = 7
        seq_len = 3

        attention = Attention(enc_output_size=enc_output_size, dec_output_size=dec_output_size)
        enc_outputs = torch.randn(seq_len, batch_size, enc_output_size)
        dec_output = torch.randn(batch_size, dec_output_size)

        output, weights = attention(enc_outputs, dec_output)
        self.assertListEqual(list(output.size()), [batch_size, enc_output_size+dec_output_size])
        self.assertListEqual(list(weights.size()), [batch_size, seq_len])


if __name__ == '__main__':
    unittest.main()


