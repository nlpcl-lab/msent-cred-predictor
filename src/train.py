import os
import tqdm
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from pytorch_transformers import (BertConfig, BertTokenizer)

import models

from utils import get_docnames
from utils import pretty_print
from utils import str2bool

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--docs', type=str, default='../data/docs')
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--bert_type', type=str,
                    default='bert-base-multilingual-cased')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--log', action='store_true')


def get_features(args, mode):
    fname = os.path.join('../cache',
                         'cached_%s_%s.pkl' % (args.bert_type, mode))
    features = models.Features(args.bert_type)

    if os.path.exists(fname):
        logger.info('Loading features from cached file %s' % fname)
        features.load_from_cache(fname)
    else:
        logger.info('Building cached feature file %s' % fname)

        dlist = get_docnames(args.docs)
        docs = [models.Document(x) for x in dlist]

        if mode == 'train':
            docs = docs[:int(len(docs) * 0.8)]
        else:
            docs = docs[int(len(docs) * 0.8):]

        features.load_from_docs(docs)
        features.save_to_cache(fname)

    return features


if __name__ == '__main__':
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(args.cuda, torch.cuda.is_available(), device)

    loss = torch.nn.MSELoss()

    config = BertConfig.from_pretrained(args.bert_type)

    model = models.CredPredictor(config)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.zero_grad()

    if args.log:
        tb_writer = SummaryWriter()

    features = get_features(args, 'train')
    test_features = get_features(args, 'test')

    logger.info('Training...')

    for _ in tqdm.trange(args.epoch, desc='Epoch'):
        for docid in tqdm.trange(features.num_of_docs(), desc='Docs'):
            div = features.num_of_rows(docid) - 2
            loss_per_doc = 0.
            for sid in tqdm.tqdm(range(1, features.num_of_rows(docid) - 1), desc='Rows'):
                prevs, aftrs, label = features.get(docid, sid, device)
                if label == 0.0:
                    div -= 1
                    continue
                model.train()

                out = loss(model(prevs, aftrs),
                           torch.tensor([label]).to(device))
                loss_per_doc += out
                optim.zero_grad()
                out.backward()
                optim.step()
            loss_per_doc /= div

            if args.log:
                tb_writer.add_scalar('loss', loss_per_doc,
                                     _ * features.num_of_docs() + docid)

        div = 0
        mape = 0.
        with torch.no_grad():
            for docid in tqdm.trange(features.num_of_docs(), desc='Test'):
                div += features.num_of_rows(docid) - 2
                for sid in range(1, features.num_of_rows(docid) - 1):
                    prevs, aftrs, label = features.get(docid, sid, device)
                    if label == 0.0:
                        div -= 1
                        continue

                    out = model(prevs, aftrs).to('cpu').item()
                    mape += abs((label - out) / label)
            mape = mape * 100 / div
            if args.log:
                tb_writer.add_scalar('mape', mape, _)
    if args.log:
        tb_writer.close()

    logger.info('Testing...')

    with torch.no_grad():
        f = open('res.tsv', 'w')
        f.write('docid\tsid\tpred\tlabel\n')

        div = 0
        mape = 0.
        for docid in tqdm.trange(test_features.num_of_docs(), desc='Test'):
            div += test_features.num_of_rows(docid) - 2
            for sid in range(1, test_features.num_of_rows(docid) - 1):
                prevs, aftrs, label = test_features.get(docid, sid, device)
                if label == 0.0:
                    div -= 1
                    continue

                out = model(prevs, aftrs).to('cpu').item()
                mape += abs((label - out) / label)

                f.write('%d\t%d\t%f\t%f\n' % (docid, sid, out, label))

            mape = mape * 100 / div
        logger.info('\nMean Absolute Percentage Error: %f' % mape)
        f.close()
