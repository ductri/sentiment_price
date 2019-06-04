import logging
import os
from datetime import datetime

import torch


class TrainingChecker:

    def __init__(self, model, dir_checkpoint, init_score):
        """
        Higher is better
        :param model:
        :param dir_checkpoint:
        :param init_score:
        """
        self._model = model
        self._dir_checkpoint = os.path.join(dir_checkpoint, datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S'))
        self._score = init_score
        self._step = None

        if not os.path.exists(self._dir_checkpoint):
            logging.warning('Path %s does not exist, auto created !' % self._dir_checkpoint)
            os.makedirs(self._dir_checkpoint)

    def update(self, score, step):
        if score > self._score:
            self._score = score
            self._step = step
            self.save_model()
            logging.info('New best score: %s', self._score)
            logging.info('Saved model at %s', self._dir_checkpoint)

    def best(self):
        return self._score, self._step

    def save_model(self):
        file_name = os.path.join(self._dir_checkpoint, '%s_%s.pt' % (self._model.__class__.__name__, self._step))
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'encoder_optimizer': self._model.encoder_optimizer.state_dict(),
            'decoder_optimizer': self._model.decoder_optimizer.state_dict()}, file_name)

    def clear_checkpoints(self):
        if len(os.listdir(self._dir_checkpoint)) == 1:
            os.rmdir(self._dir_checkpoint)
        else:
            raise Exception('There are more than 1 file in checkpoint directory, check it !!!')


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
