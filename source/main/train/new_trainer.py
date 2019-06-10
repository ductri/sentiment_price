import logging
import time
import itertools
import copy

import torch
import numpy as np
from sklearn import metrics

from naruto_skills.dl_logging import DLTBHandler, DLLoggingHandler, DLLogger


class TrainingLoop:
    def __init__(self, model, train_loader, device, num_epoch, train_logger, evaluator):
        """
        Only need to re-implement function `step`
        :param model:
        :param train_loader:
        :param device:
        :param num_epoch:
        :param train_logger:
        :param evaluator:
        """
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.num_epoch = num_epoch
        self.train_logger = train_logger
        self.evaluator = evaluator

        # All metrics are measured by batch
        self.__current_state = {'step': 0, 'duration': None, 'loss': None, 'inputs': None}

    def run(self):
        logging.info('----------------------- START TRAINING LOOP -----------------------')
        for _ in range(self.num_epoch):
            for inputs in self.train_loader:
                inputs = self.__move_input_to_device(inputs)
                self.step(inputs)

                self.train_logger.run(self.__current_state)
                self.evaluator.run(self.__current_state['step'])

        self.train_logger.close()
        self.evaluator.close()

    def step(self, inputs):
        """

        :param inputs:
        :return: tuple (loss, duration)
        """
        start = time.time()
        self.model.train()
        step_loss = self.model.train_batch(inputs[0], inputs[2])

        self.__current_state['step'] += 1
        self.__current_state['duration'] = time.time() - start
        self.__current_state['loss'] = step_loss
        self.__current_state['inputs'] = inputs

    def __move_input_to_device(self, inputs):
        inputs = [i.to(self.device) for i in inputs]
        return inputs


class Evaluator:
    def __init__(self, model, eval_loader, device, interval, eval_logger, checker):
        """
        Responsibilities:
        - What to measure
        - When to measure
        - Where to store the measured metrics

        Only need to re-implement 2 functions:
        - update_current_state
        - step
        This class will be in charge on getting all necessary metrics on evaluation data set.
        :param model:
        :param eval_loader:
        :param device:
        :param interval:
        """

        self.model = model
        self.eval_loader = eval_loader
        self.device = device
        self.interval = interval
        self.eval_logger = eval_logger
        self.checker = checker

        self.__registered_functions = []

    def step(self, inputs):
        """
        :param inputs:
        :return:
        """
        loss_np = self.model.get_loss(inputs[0], inputs[2]).cpu().item()
        prob_tensor = self.model(*inputs)
        prob_np = prob_tensor.cpu().numpy()[:, 1]
        true_np = inputs[2].cpu().numpy()
        return [loss_np], prob_np, true_np

    def update_current_state(self, measures):
        """

        :param measures: Tuple with len(measures) == number_of_metrics
        :return:
        """
        # All metrics are measured by the whole evaluation set
        current_state = {'loss': None, 'loss_std': None, 'P': None, 'R': None, 'F1': None, 'AUC': None}

        losses_np, probs_np, trues_np = measures
        current_state['loss'] = np.mean(losses_np)
        current_state['loss_std'] = np.mean(losses_np)
        current_state['P'] = metrics.precision_score(y_true=trues_np, y_pred=probs_np >= 0.5)
        current_state['R'] = metrics.recall_score(y_true=trues_np, y_pred=probs_np >= 0.5)
        current_state['F1'] = metrics.f1_score(y_true=trues_np, y_pred=probs_np >= 0.5)
        current_state['AUC'] = metrics.roc_auc_score(y_true=trues_np, y_score=probs_np)
        return current_state

    def run(self, global_step):
        if global_step % self.interval == 0:
            self.__main_run(global_step)

    def __main_run(self, global_step):
        self.model.eval()
        with torch.no_grad():
            start = time.time()
            measures = []
            for inputs in self.eval_loader:
                inputs = self.__move_input_to_device(inputs)
                batch_metrics = self.step(inputs)
                measures.append(batch_metrics)
            measures = list(zip(*measures))
            for i in range(len(measures)):
                measures[i] = list(itertools.chain(*measures[i]))
                measures[i] = np.array(measures[i])

            current_state = self.update_current_state(measures)
            current_state['step'] = global_step
            current_state['duration'] = time.time() - start
            self.eval_logger.run(current_state)
            self.checker.update(-current_state['loss'], current_state['step'])

    def __move_input_to_device(self, inputs):
        inputs = [i.to(self.device) for i in inputs]
        return inputs

    def close(self):
        self.eval_logger.close()


class TrainingLogger:
    def __init__(self, model, measure_interval, predict_interval, path_to_file, input_transform, output_transform):
        """
        Responsibilities:
        - What will be logged
        - Where will be logged (params)
        - When will be logged (params)
        These 3 tasks can be divided, but I put them all here for fast prototype. I also don't think these things need
        more flexible.
        :param model --> what to log
        :param measure_interval --> when to log metrics
        :param predict_interval --> when to log predictions
        :param path_to_file --> where to log
        :param input_transform
        :param output_transform
        """
        self.model = model
        self.measure_interval = measure_interval
        self.predict_interval = predict_interval

        self.my_logger = DLLogger()
        self.my_logger.add_handler(DLLoggingHandler())
        self.my_logger.add_handler(DLTBHandler(path_to_file))

        self.input_transform = input_transform
        self.output_transform = output_transform

        # We only do streaming loss because loss figure is calculated every steps, no matter our want
        self.__current_state = {'streaming_loss': []}

        self.tag_loss = 'loss'
        self.tag_p = 'P'
        self.tag_r = 'R'
        self.tag_duration = 'duration'

    def run(self, current_state):
        self.model.eval()
        with torch.no_grad():
            self.log_metrics(current_state)
            self.log_prediction(current_state)

    def log_metrics(self, current_state):
        def measure(inputs):
            self.model.eval()
            prob_predict = self.model(*inputs)
            prob_np = prob_predict.cpu().numpy()
            predict_np = np.argmax(prob_np, axis=1)
            target_np = inputs[2].cpu().numpy()

            train_precision = metrics.precision_score(y_true=target_np, y_pred=predict_np, average='macro')
            train_recall = metrics.recall_score(y_true=target_np, y_pred=predict_np, average='macro')
            return train_precision, train_recall

        if current_state['step'] % self.measure_interval == 0:
            inputs = current_state['inputs']
            global_step = current_state['step']

            p, r = measure(inputs)
            streaming_loss = np.mean(self.__current_state['streaming_loss'])

            self.my_logger.add_scalar(self.tag_loss, streaming_loss, global_step)
            self.my_logger.add_scalar(self.tag_p, p, global_step)
            self.my_logger.add_scalar(self.tag_r, r, global_step)
            self.my_logger.add_scalar(self.tag_duration, current_state['duration'], global_step)

            self.__current_state['streaming_loss'].clear()

        self.__current_state['streaming_loss'].append(current_state['loss'])

    def log_prediction(self, current_state):
        if current_state['step'] % self.predict_interval == 0:
            sample_size = 5
            inputs = current_state['inputs']
            input_tensors = [input_tensor[:sample_size] for input_tensor in inputs]
            predict_tensor = self.model(*input_tensors)

            input_transformed = self.input_transform(input_tensors[0].cpu().numpy())
            target_transformed = self.output_transform(input_tensors[2].cpu().numpy())
            predict_transformed = self.output_transform(predict_tensor.cpu().numpy())
            logging.info('----------- PREDICT FROM TRAIN -----------')
            for idx, (src, pred, tgt) in enumerate(zip(input_transformed, predict_transformed, target_transformed)):
                logging.info('Sample %s ', idx + 1)
                logging.info('Source:\t%s', src)
                logging.info('Predict:\t%s', pred)
                logging.info('Target:\t%s', tgt)
                logging.info('------')

    def close(self):
        self.my_logger.close()


class EvaluateLogger:
    def __init__(self, path_to_file):
        """
        Responsibilities:
        - Just log whatever Evaluator measures
        :param path_to_file:
        """
        self.my_logger = DLLogger()
        self.my_logger.add_handler(DLLoggingHandler())
        self.my_logger.add_handler(DLTBHandler(path_to_file))

    def run(self, current_state):
        logging.info('----------------------- LOGGING FROM EVALUATION -----------------------')
        current_state = copy.deepcopy(current_state)
        step = current_state['step']
        del current_state['step']
        data_to_log = [('%s' % k, v) for k, v in current_state.items()]
        for tag_name, value in data_to_log:
            self.my_logger.add_scalar(tag_name, value, step)

    def close(self):
        self.my_logger.close()