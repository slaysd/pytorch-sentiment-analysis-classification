import os

import torch
from torchtext.data import Field, LabelField, TabularDataset, Iterator
from torchtext.vocab import Vectors

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


class MyDataset(object):

    def __init__(self, root_dir='data', batch_size=64, use_vector=True):
        self.TEXT = Field(sequential=True, use_vocab=True,
                          tokenize='spacy', lower=True, batch_first=True)
        self.LABEL = LabelField(tensor_type=torch.FloatTensor)
        vectors = Vectors(name='mr_vocab.txt', cache='./')

        dataset_path = os.path.join(root_dir, '{}.tsv')
        self.dataset = {}
        self.dataloader = {}
        for target in ['train', 'dev', 'test']:
            self.dataset[target] = TabularDataset(
                path=dataset_path.format(target),
                format='tsv',
                fields=[('text', self.TEXT), ('label', self.LABEL)]
            )
            if use_vector:
                self.TEXT.build_vocab(self.dataset[target], max_size=25000, vectors=vectors)
            else:
                self.TEXT.build_vocab(self.dataset[target], max_size=25000)

            self.LABEL.build_vocab(self.dataset[target])
            self.dataloader[target] = Iterator(self.dataset[target],
                                               batch_size=batch_size,
                                               device=None,
                                               repeat=False,
                                               sort_key=lambda x: len(x.text),
                                               shuffle=True)


if __name__ == '__main__':
    dataset = MyDataset()
