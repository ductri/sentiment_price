import os
import time
import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
import torch

from utils import metrics
from utils.training_checker import TrainingChecker


class MyTrainingChecker(TrainingChecker):
    def __init__(self, model, dir_checkpoint, init_score):
        super(MyTrainingChecker, self).__init__(model, dir_checkpoint + '/' + model.__class__.__name__, init_score)

    def save_model(self):
        file_name = os.path.join(self._dir_checkpoint, '%s.pt' % self._step)
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer': self._model.optimizer.state_dict(),
            'step': self._step,
            'best_score': self._score
        }, file_name)


def train(model, train_loader, eval_loader, dir_checkpoint, device, num_epoch=10, print_every=1000, predict_every=500,
          eval_every=500, input_transform=None, output_transform=None, init_step=0):
    if input_transform is None:
        input_transform = lambda *x: x
    if output_transform is None:
        output_transform = lambda *x: x

    def predict_and_print_sample(*inputs):
        sample_size = 5
        input_tensors = [input_tensor[:sample_size] for input_tensor in inputs]
        predict_tensor = model(input_tensors[0])

        input_transformed = input_transform(input_tensors[0].cpu().numpy())
        target_transformed = output_transform(input_tensors[1].cpu().numpy())

        seq_len_np = input_tensors[2].cpu().numpy()
        predict_transformed = output_transform(predict_tensor.cpu().numpy())
        predict_transformed = [' '.join(doc.split()[:seq_len_np[idx]]) for idx, doc in enumerate(predict_transformed)]

        for idx, (src, pred, tgt) in enumerate(zip(input_transformed, predict_transformed, target_transformed)):
            logging.info('Sample %s ', idx + 1)
            logging.info('Source:\t%s', src)
            logging.info('Predict:\t%s', pred)
            logging.info('Target:\t%s', tgt)
            logging.info('------')

    t_loss_tracking = metrics.MeanMetrics()
    e_w_a_tracking = metrics.MeanMetrics()
    e_s_a_tracking = metrics.MeanMetrics()
    training_checker = MyTrainingChecker(model, dir_checkpoint, init_score=0)

    step = init_step
    model.to(device)
    logging.info('----------------------- START TRAINING -----------------------')
    for _ in range(num_epoch):
        for inputs in train_loader:
            inputs = [i.to(device) for i in inputs]
            start = time.time()
            train_loss = model.train_batch(*inputs)
            t_loss_tracking.add(train_loss)
            step += 1
            with torch.no_grad():
                if step % print_every == 0 or step == 1:
                    model.eval()
                    prediction_numpy = model(inputs[0]).cpu().numpy()
                    target_numpy = inputs[1].cpu().numpy()
                    seq_len_numpy = inputs[2].cpu().numpy()
                    w_acc = cal_word_acc(prediction_numpy, target_numpy, seq_len_numpy)
                    s_acc = cal_sen_acc(prediction_numpy, target_numpy, seq_len_numpy)

                    logging.info(
                        'Step: %s \t L_mean: %.4f±%.4f \t w_a: %.4f \t s_a: %.4f \t Duration: %.4f s/step' % (
                            step, t_loss_tracking.mean(), float(np.std(t_loss_tracking.figures)),
                            w_acc, s_acc, time.time() - start))
                    t_loss_tracking.reset()

                if step % predict_every == 0:
                    model.eval()

                    logging.info('\n\n------------------ Predict samples from train ------------------ ')
                    logging.info('Step: %s', step)
                    predict_and_print_sample(*inputs)

                if step % eval_every == 0:
                    model.eval()
                    e_w_a_tracking.reset()
                    e_s_a_tracking.reset()

                    start = time.time()
                    for eval_inputs in eval_loader:
                        eval_inputs = [i.to(device) for i in eval_inputs]
                        e_pred_numpy = model(eval_inputs[0]).cpu().numpy()
                        e_target_numpy = eval_inputs[1].cpu().numpy()
                        e_seq_len_numpy = eval_inputs[2].cpu().numpy()
                        e_w_a_tracking.add(cal_word_acc(e_pred_numpy, e_target_numpy, e_seq_len_numpy))
                        e_s_a_tracking.add(cal_sen_acc(e_pred_numpy, e_target_numpy, e_seq_len_numpy))
                    logging.info('\n\n------------------ \tEvaluation\t------------------')
                    logging.info('Step: %s', step)
                    logging.info('Number of batchs: %s', e_w_a_tracking.get_count())
                    logging.info('w_a: %.4f±%.4f \t s_a: %.4f±%.4f \t Duration: %.4f s/step',
                                 e_w_a_tracking.mean(), float(np.std(e_w_a_tracking.figures)),
                                 e_s_a_tracking.mean(), float(np.std(e_s_a_tracking.figures)),
                                 time.time() - start)

                    training_checker.update(e_w_a_tracking.mean(), step)
                    best_score, best_score_step = training_checker.best()
                    logging.info('Current best score: %s recorded at step %s', best_score, best_score_step)

                    eval_inputs = next(iter(eval_loader))
                    eval_inputs = [item.to(device) for item in eval_inputs]
                    predict_and_print_sample(*eval_inputs)


