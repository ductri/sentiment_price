import unittest
import torch

from model_def.deprecated.seq2seq_attn_with_src import Seq2SeqAttnWithSrc


class TestSeq2SeqAttnWithSrc(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward(self):
        src_vocab_size = 5
        batch_size = 2
        seq_len = 10

        inputs_idx = torch.randint(src_vocab_size, size=(batch_size, seq_len))
        model = Seq2SeqAttnWithSrc(src_vocab_size=src_vocab_size, tgt_vocab_size=5, start_idx=1, end_idx=2)
        output = model(inputs_idx)
        self.assertEqual(output.size(), (batch_size, seq_len))

    def test_train(self):
        src_vocab_size = 7
        tgt_vocab_size = 7
        batch_size = 5
        seq_length = 100
        end_idx = 6
        inputs_idx = torch.randint(low=0, high=src_vocab_size-2, size=(batch_size, seq_length)).to(self.device)
        target_idx = inputs_idx.clone()
        length = torch.mul(torch.ones(batch_size), 5).to(self.device)

        model = Seq2SeqAttnWithSrc(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, start_idx=5,
                                   end_idx=end_idx)
        model.train()
        model.to(self.device)
        model.lr_rate = 1e-3
        for step in range(100):
            model.train()
            loss = model.train_batch(inputs_idx, target_idx, length)
            print('Step: %s - Loss: %.4f' % (step, loss))

        model.eval()
        pred = model(inputs_idx)
        pred_np = pred.int().cpu().numpy()
        length_np = length.int().cpu().numpy()
        target_idx_np = target_idx.int().cpu().numpy()

        pred_list = []
        for i, l in enumerate(length_np):
            pred_list.extend(pred_np[i, :l])
        target_list = []
        for i, l in enumerate(length_np):
            target_list.extend(target_idx_np[i, :l])

        self.assertListEqual(pred_list, target_list)


if __name__ == '__main__':
    unittest.main()


