import os
import logging
from datetime import datetime

import torch
from naruto_skills.training_checker import TrainingChecker

from data_for_train import is_question as my_dataset
from model_def.lstm_attention import LSTMAttention
from utils import pytorch_utils
from train.new_trainer import TrainingLoop, TrainingLogger, EvaluateLogger, Evaluator


def input2_text(first_input, *params):
    return my_dataset.voc.idx2docs(first_input)


def target2_text(first_input, *params):
    return first_input


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    BATCH_SIZE = 128
    NUM_EPOCHS = 500
    NUM_WORKERS = 0
    PRINT_EVERY = 100
    PREDICT_EVERY = 500
    EVAL_EVERY = 500
    PRE_TRAINED_MODEL = ''

    my_dataset.bootstrap()
    train_loader = my_dataset.get_dl_train(batch_size=BATCH_SIZE, size=None)
    eval_loader = my_dataset.get_dl_eval(batch_size=BATCH_SIZE, size=None)
    logging.info('There will be %s steps for training', NUM_EPOCHS * len(train_loader))
    model = LSTMAttention(vocab_size=len(my_dataset.voc.index2word), no_class=2)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info('Model architecture: \n%s', model)
    logging.info('Total trainable parameters: %s', pytorch_utils.count_parameters(model))

    init_step = 0
    # Restore model
    if PRE_TRAINED_MODEL != '':
        checkpoint = torch.load(PRE_TRAINED_MODEL, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        init_step = checkpoint.get('step', 0)

        logging.info('Load pre-trained model from %s successfully', PRE_TRAINED_MODEL)

    root_dir = '/source/main/train/output/'
    exp_id = datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S')

    path_checkpoints = os.path.join(root_dir, 'saved_models', model.__class__.__name__, exp_id)
    training_checker = TrainingChecker(model, root_dir=path_checkpoints, init_score=-10000)

    path_logging = os.path.join(root_dir, 'logging', model.__class__.__name__, exp_id)
    train_logger = TrainingLogger(model, measure_interval=PRINT_EVERY, predict_interval=PREDICT_EVERY,
                                     path_to_file=path_logging + '_train', input_transform=input2_text,
                                     output_transform=target2_text)

    eval_logger = EvaluateLogger(path_logging + '_validate')
    evaluator = Evaluator(model, eval_loader, device, EVAL_EVERY, eval_logger, training_checker)

    training_loop = TrainingLoop(model, train_loader, device, NUM_EPOCHS, train_logger, evaluator)
    training_loop.run()
