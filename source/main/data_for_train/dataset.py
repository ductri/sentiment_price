import logging

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch

from naruto_skills.voc import Voc

voc = None
MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'


class Docs(Dataset):
    def __init__(self, path_to_file, size=None):
        super(Docs, self).__init__()
        df = pd.read_csv(path_to_file)
        data = df[['mention', 'label']]
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)

        size = size or data.shape[0]
        data = data.sample(size)

        self.mention = list(data['mention'])
        self.label = list(data['label'])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        mention = self.mention[idx]
        label = self.label[idx]
        word_length = len(mention.split())

        mention_idx = voc.docs2idx([mention], equal_length=MAX_LENGTH)[0]

        return mention_idx, word_length, label


def bootstrap():
    global voc
    path_src = ROOT + 'main/vocab/output/voc.pkl'
    voc = Voc.load(path_src)

    logging.info('Vocab from file %s contains %s tokens', path_src, len(voc.index2word))


def create_data_loader(path_to_csv, batch_size, num_workers, size=None, shuffle=True):
    def collate_fn(list_data):
        """
        shape == (batch_size, col1, col2, ...)
        """
        data = zip(*list_data)
        data = [np.stack(col, axis=0) for col in data]
        data = [torch.from_numpy(col) for col in data]
        return data

    my_dataset = Docs(path_to_csv, size=size)
    logging.info('Data at %s contains %s samples', path_to_csv, len(my_dataset))
    dl = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dl


def get_dl_train(batch_size, size=None):
    return create_data_loader(ROOT + 'main/data_for_train/output/is_price/train.csv', batch_size,
                              NUM_WORKERS, size=size, shuffle=True)


def get_dl_test(batch_size):
    return create_data_loader(ROOT + 'main/data_for_train/output/is_price/test.csv', batch_size,
                              NUM_WORKERS, shuffle=False)


def get_dl_eval(batch_size, size=None):
    return create_data_loader(ROOT + 'main/data_for_train/output/is_price/eval.csv', batch_size,
                              NUM_WORKERS, shuffle=False, size=size)