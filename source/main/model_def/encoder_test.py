import unittest
import torch

from model_def.encoder import Encoder


class TestEncoder(unittest.TestCase):

    def test_encoder(self):

        docs = torch.Tensor([[1, 2, 3, 4], [1, 2, 2, 4]]).long()
        batch_size = docs.size(0)

        encoder = Encoder(vocab_size=5)
        _, (h_n, c_n) = encoder(docs)

        self.assertEqual(h_n.shape, (6, batch_size, 512))
        self.assertEqual(c_n.shape, (6, batch_size, 512))

    def test_encoder_step_by_step(self):
        batch_size = 2
        seq_len = 2
        vocab_size = 100
        docs = torch.randint(vocab_size, size=(batch_size, seq_len))

        encoder = Encoder(vocab_size=vocab_size, is_bidirectional=False)
        encoder.eval()
        with torch.no_grad():
            _, (h_n_1, c_n_1) = encoder(docs)

            _, (h_n_2, c_n_2) = encoder(docs[:, 0:1])
            for step in range(1, seq_len):
                _, (h_n_2, c_n_2) = encoder(docs[:, step:step+1], (h_n_2, c_n_2))

            self.assertEqual(torch.norm(h_n_1 - h_n_2), 0)
            self.assertEqual(torch.norm(c_n_1 - c_n_2), 0)


if __name__ == '__main__':
    unittest.main()
