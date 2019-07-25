import torch
import spacy

from torch.utils.data import Dataset

MAX_SENTENCE_LENGTH = 256
spacy.load('en')


class Row:
    def __init__(self, line):
        'Line: (No, Body, Score)'
        self.no, self.body, self.score = line.split('\t')


class Document:
    def __init__(self, dname):
        self.id = dname
        self.rows = list()

        f = open(path, 'r', encoding='UTF-8')
        for line in f:
            self.rows.append(Row(line))
        f.close()


class ESIMDataset(Dataset):
    'Dataset class for adopted ESIM(for the first part)'

    def __init__(self, docs):
        raise NotImplementedError
