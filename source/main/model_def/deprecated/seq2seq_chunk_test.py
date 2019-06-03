import unittest
import torch

from model_def.deprecated.seq2seq_chunk import Seq2SeqChunk


class TestSeq2SeqAttn(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward(self):
        src_vocab_size = 5
        batch_size = 2
        seq_len = 101

        inputs_idx = torch.randint(src_vocab_size, size=(batch_size, seq_len))
        model = Seq2SeqChunk(src_vocab_size=src_vocab_size, tgt_vocab_size=5, start_idx=1, padding_idx=3)
        output = model(inputs_idx)
        self.assertEqual(output.size(), (batch_size, seq_len))

    def test_train(self):
        src_vocab_size = 7
        tgt_vocab_size = 7
        batch_size = 2
        max_length = 10
        inputs_idx = torch.randint(low=0, high=src_vocab_size-2, size=(batch_size, max_length)).to(self.device)
        target_idx = torch.randint(low=0, high=src_vocab_size - 2, size=(batch_size, max_length)).to(self.device)
        # target_idx = inputs_idx.clone()
        print('Input', inputs_idx)
        print('Target', target_idx)
        # length = torch.randint(low=3, high=10, size=(batch_size, )).to(self.device)
        length = torch.tensor([20, 27]).to(self.device)
        model = Seq2SeqChunk(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, start_idx=5, padding_idx=6,
                             max_length=max_length)
        model.train()
        model.to(self.device)
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


