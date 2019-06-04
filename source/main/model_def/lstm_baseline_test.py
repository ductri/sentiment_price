import unittest
import torch

from model_def.lstm_baseline import LSTMBaseline


class TestLSTMBaseline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward(self):
        vocab_size = 5
        batch_size = 7
        no_class = 2
        seq_len = 10

        inputs_idx = torch.randint(vocab_size, size=(batch_size, seq_len))
        model = LSTMBaseline(vocab_size=vocab_size, no_class=no_class)
        output = model(inputs_idx)
        self.assertEqual(output.size(), (batch_size, no_class))

    def test_train(self):
        vocab_size = 7
        batch_size = 5
        max_length = 100
        no_class = 2

        inputs_idx = torch.randint(low=0, high=vocab_size, size=(batch_size, max_length)).to(self.device)
        target = torch.randint(no_class, size=(batch_size,)).to(self.device)

        model = LSTMBaseline(vocab_size=vocab_size, no_class=no_class)
        model.train()
        model.to(self.device)
        for step in range(7):
            model.train()
            loss = model.train_batch(inputs_idx, target)
            print('Step: %s - Loss: %.4f' % (step, loss))

        model.eval()
        predict = model(inputs_idx)
        predict = torch.argmax(predict, dim=1)

        print('Target', target)
        print('Predict', predict)

        self.assertListEqual(list(target), list(predict))


if __name__ == '__main__':
    unittest.main()


