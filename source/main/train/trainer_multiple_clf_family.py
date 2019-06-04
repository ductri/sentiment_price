import os
import logging
import time

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report, f1_score, \
    average_precision_score
import torch

from utils.training_checker import MyTrainingChecker
from utils import metrics


def __eval(model, data_loader, device, extra_logging=False):
    logging.info('\n\n')
    logging.info('\t------------------ \tBegin Evaluation\t------------------')
    loss_tracking = metrics.MeanMetrics()
    probs_np = []
    trues_np = []
    start = time.time()

    model.eval()
    for idx, inputs in enumerate(data_loader):
        inputs = [i.to(device) for i in inputs]
        loss_np = model.get_loss(inputs[0], inputs[2])
        loss_tracking.add(loss_np)

        prob_tensor = model(*inputs)
        probs_np.extend(prob_tensor.cpu().numpy())
        trues_np.extend(inputs[2].cpu().numpy())
    probs_np = np.array(probs_np)
    trues_np = np.array(trues_np)
    logging.info('Number of batchs: %s', loss_tracking.get_count())
    preds_np = np.argmax(probs_np, axis=1)

    logging.info('L_mean: %.4f±%.4f \t \t P: %.4f±%.4f \t R: %.4f±%.4f \t F1: %.4f±%.4f '
                 '\t Duration: %.3f s/step' %
                     (loss_tracking.mean(), float(np.std(loss_tracking._figures)),
                      precision_score(y_true=trues_np, y_pred=preds_np, average='macro'), 0,
                      recall_score(y_true=trues_np, y_pred=preds_np, average='macro'), 0,
                      f1_score(y_true=trues_np, y_pred=preds_np, average='macro'), 0,
                      time.time() - start))
    if extra_logging:
        logging.info('\n %s', classification_report(trues_np, preds_np, digits=4))
    logging.info('\t------------------ \tEnd Evaluation\t------------------\n\n')
    return -loss_tracking.mean()


def train(model, train_loader, eval_loader, dir_checkpoint, device, num_epoch=10, print_every=1000, predict_every=500,
          eval_every=500, input_transform=None, output_transform=None, test_loader=None):
    if input_transform is None:
        input_transform = lambda *x: x
    if output_transform is None:
        output_transform = lambda *x: x

    def predict_and_print_sample(tensors):
        sample_size = 5
        tensors = [input_tensor[:sample_size] for input_tensor in tensors]
        prob_predict = model(*tensors)
        prob_predict_np = prob_predict.cpu.numpy()
        predict_np = torch.argmax(prob_predict, dim=1).cpu().numpy()
        target_np = tensors[2].cpu().numpy()
        word_input_np = tensors[0].cpu().numpy()

        input_transformed = input_transform(word_input_np)
        predict_transformed = output_transform(predict_np)
        target_transformed = output_transform(target_np)

        for idx, (src, pred, prob, tgt) in enumerate(zip(input_transformed, predict_transformed, prob_predict_np, target_transformed)):
            logging.info('Sample %s ', idx + 1)
            logging.info('Source:\t%s', src)
            logging.info('Predict:\t%s', pred)
            logging.info('Prob:\t%s', prob)
            logging.info('Target:\t%s', tgt)
            logging.info('------\n')

    t_loss_tracking = metrics.MeanMetrics()
    e_loss = metrics.MeanMetrics()
    e_precision = metrics.MeanMetrics()
    e_recall = metrics.MeanMetrics()
    e_auc = metrics.MeanMetrics()
    training_checker = MyTrainingChecker(model, dir_checkpoint, init_score=-1000)

    step = 0
    model.to(device)
    logging.info('----------------------- START TRAINING -----------------------')
    for _ in range(num_epoch):
        for inputs in train_loader:
            inputs = [i.to(device) for i in inputs]
            start = time.time()
            train_loss = model.train_batch(inputs[:2], inputs[2])
            t_loss_tracking.add(train_loss)
            step += 1
            with torch.no_grad():
                if step % print_every == 0 or step == 1:
                    model.eval()
                    prob_predict = model(*inputs)
                    predict_np = torch.argmax(prob_predict, dim=1).cpu().numpy()
                    target_np = inputs[2].cpu().numpy()

                    train_precision = precision_score(y_true=target_np, y_pred=predict_np, average='macro')
                    train_recall = recall_score(y_true=target_np, y_pred=predict_np, average='macro')
                    logging.info(
                        'Step: %s \t L: %.4f±%.4f \t P: %.4f±%.4f \t R: %.4f±%.4f \t Duration: %.3f s/step' % (
                            step, t_loss_tracking.mean(), float(np.std(t_loss_tracking._figures)),
                            train_precision, 0,
                            train_recall, 0,
                            time.time() - start))
                    t_loss_tracking.reset()

                if step % predict_every == 0:
                    model.eval()

                    logging.info('\n\n------------------ Predict samples from train ------------------ ')
                    logging.info('Step: %s', step)
                    predict_and_print_sample(inputs)

                if step % eval_every == 0:
                    model.eval()
                    e_loss.reset()
                    e_precision.reset()
                    e_recall.reset()
                    e_auc.reset()

                    logging.info('Step: %s', step)
                    score = __eval(model, data_loader=eval_loader, device=device)

                    training_checker.update(score, step)
                    best_score, best_score_step = training_checker.best()
                    logging.info('Current best score: %s recorded at step %s', best_score, best_score_step)

    if test_loader is not None:
        logging.info('----------------------- START TESTING -----------------------')
        path_to_best_model = training_checker.get_path_to_best()
        logging.info('Load best model from %s', path_to_best_model)
        checkpoint = torch.load(path_to_best_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        with torch.no_grad():
            __eval(model, test_loader, device, extra_logging=True)

    return training_checker.get_path_to_best()

