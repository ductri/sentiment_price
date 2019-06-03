import logging
import torch

from data_for_train import dataset as my_dataset
from model_def.deprecated.seq2seq_attn_with_src import Seq2SeqAttnWithSrc
from utils import pytorch_utils
from train.trainer import train


def input2_text(first_input, *params):
    return my_dataset.voc_src.idx2docs(first_input)


def target2_text(first_input, *params):
    return my_dataset.voc_tgt.idx2docs(first_input)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    BATCH_SIZE = 32
    NUM_EPOCHS = 500
    NUM_WORKERS = 1
    PRINT_EVERY = 100
    PREDICT_EVERY = 5000
    EVAL_EVERY = 10000
    PRE_TRAINED_MODEL = '/source/main/train/output/saved_models/Seq2SeqAttnWithSrc/2019-06-01T08:51:02/90000.pt'

    my_dataset.bootstrap()
    train_loader = my_dataset.get_dl_train(batch_size=BATCH_SIZE, size=None)
    eval_loader = my_dataset.get_dl_eval(batch_size=BATCH_SIZE, size=None)
    logging.info('There will be %s steps for training', NUM_EPOCHS * (len(train_loader)/BATCH_SIZE))
    model = Seq2SeqAttnWithSrc(src_vocab_size=len(my_dataset.voc_src.index2word),
                    tgt_vocab_size=len(my_dataset.voc_tgt.index2word),
                    start_idx=2,
                    end_idx=3
                    )
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

    train(model, train_loader, eval_loader, dir_checkpoint='/source/main/train/output/saved_models/', device=device,
          num_epoch=NUM_EPOCHS, print_every=PRINT_EVERY, predict_every=PREDICT_EVERY, eval_every=EVAL_EVERY,
          input_transform=input2_text, output_transform=target2_text, init_step=init_step)
