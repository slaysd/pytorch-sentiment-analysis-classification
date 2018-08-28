import os

import torch
from torchtext.data import Field, LabelField, TabularDataset, Iterator


class MyDataset(object):

    def __init__(self, root_dir='data', batch_size=64):
        self.TEXT = Field(sequential=True, use_vocab=True,
                          tokenize=str.split, lower=True, batch_first=True)
        self.LABEL = LabelField(tensor_type=torch.FloatTensor)

        dataset_path = os.path.join(root_dir, '{}.tsv')
        self.dataset = {}
        self.dataloader = {}
        for target in ['train', 'dev', 'test']:
            self.dataset[target] = TabularDataset(
                path=dataset_path.format('train'),
                format='tsv',
                fields=[('text', self.TEXT), ('label', self.LABEL)]
            )
            self.TEXT.build_vocab(self.dataset[target], max_size=25000)
            self.LABEL.build_vocab(self.dataset[target])
            self.dataloader[target] = Iterator(self.dataset[target],
                                               batch_size=batch_size,
                                               device=None,
                                               repeat=False,
                                               shuffle=True)


if __name__ == '__main__':
    dataset = MyDataset()
