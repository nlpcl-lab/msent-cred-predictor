import torch
import spacy
import pickle
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from pytorch_transformers import (WEIGHTS_NAME, BertModel, BertTokenizer)
from utils import (truncate_seq_pair, pretty_print)

TEST_IDX = 2
MAX_SENTENCE_LENGTH = 256
spacy.load('en')

CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
PAD_TOKEN = 0


class Row:
    def __init__(self, line):
        'Line: (No, Body, Score)'
        self.no, self.body, self.score = line.strip().split('\t')
        self.score = float(self.score)


class Document:
    def __init__(self, dname):
        self.id = dname
        self.rows = list()

        f = open(dname, 'r', encoding='UTF-8')
        for line in f:
            self.rows.append(Row(line))
        f.close()


class DocAsInput:
    def __init__(self, raw_feats, scores, device):
        self.input_ids = torch.tensor(
            [f[0] for f in raw_feats], dtype=torch.long).to(device)
        self.attention_mask = torch.tensor(
            [f[1] for f in raw_feats], dtype=torch.long).to(device)
        self.token_type_ids = torch.tensor(
            [f[2] for f in raw_feats], dtype=torch.long).to(device)
        self.labels = torch.tensor(scores).view(-1, 1).to(device)


class Features:
    def __init__(self, bert_type):
        self._docs = list()
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)

    def load_from_docs(self, docs):
        for doc in docs:
            _doc = list()
            for row in doc.rows:
                _doc.append(
                    (row.no, self.tokenizer.tokenize(row.body),
                     row.score))
            self._docs.append(_doc)

    def load_from_cache(self, fname):
        with open(fname, 'rb') as f:
            self._docs = pickle.load(f)

    def save_to_cache(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self._docs, f)

    def num_of_docs(self):
        return len(self._docs)

    def num_of_rows(self, docid):
        return len(self._docs[docid])

    def get(self, docid, sid, device):
        prevs = list()
        prev_scores = list()
        aftrs = list()
        aftr_scores = list()

        for prev_sid in range(max(0, sid - TEST_IDX), sid):
            prevs.append(self._get_pair(docid, prev_sid, sid))
            prev_scores.append(self._docs[docid][prev_sid][2])

        for aftr_sid in range(sid, min(len(self._docs[docid]), sid + TEST_IDX)):
            aftrs.append(self._get_pair(docid, sid, aftr_sid))
            aftr_scores.append(self._docs[docid][aftr_sid][2])

        prevs = DocAsInput(prevs, prev_scores, device)
        aftrs = DocAsInput(aftrs, aftr_scores, device)

        return (prevs, aftrs, self._docs[docid][sid][2])

    def _get_pair(self, docid, prev_sid, aftr_sid):
        tok_a = self._docs[docid][prev_sid][1]
        tok_b = self._docs[docid][aftr_sid][1]

        la, lb = truncate_seq_pair(tok_a, tok_b,
                                   MAX_SENTENCE_LENGTH - 3)

        toks = [CLS_TOKEN] + tok_a[:la] + [SEP_TOKEN] + \
            tok_b[:lb] + [SEP_TOKEN]
        segs = [0] * (la + 2) + [1] * (lb + 1)
        inps = self.tokenizer.convert_tokens_to_ids(toks)
        inp_mask = [1] * len(inps)

        pad_len = MAX_SENTENCE_LENGTH - len(inps)
        inps += [PAD_TOKEN] * pad_len
        inp_mask += [0] * pad_len
        segs += [0] * pad_len

        return [inps, inp_mask, segs]


class CredPredictor(nn.Module):
    def __init__(self, config):
        super(CredPredictor, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.referee = nn.LSTM(2, 1)
        self.final = nn.Linear(config.hidden_size + 1, 1)

    def forward(self, prev, aftr):
        # TODO: if number of sentences is too big, process would be killed
        _, prev_pooled = self.bert(
            prev.input_ids,
            prev.token_type_ids,
            prev.attention_mask)
        _, aftr_pooled = self.bert(
            aftr.input_ids,
            aftr.token_type_ids,
            aftr.attention_mask)

        prev_pooled = self.dropout(prev_pooled)
        aftr_pooled = self.dropout(aftr_pooled)

        # print('\n\n')
        # pretty_print([('prev_pooled', prev_pooled.size()),
        #               ('aftr_pooled', aftr_pooled.size()),
        #               ('prev.labels', prev.labels.size()),
        #               ('aftr.labels', aftr.labels.size())])

        prevs = torch.cat((prev_pooled, prev.labels), 1)
        aftrs = torch.cat((aftr_pooled, aftr.labels), 1)

        # pretty_print([('prevs', prevs.size()),
        #              ('aftrs', aftrs.size())])

        # TODO: Should we think about the order of sentences?
        # TODO: torch.mean() or torch.prod()?

        prev_res = torch.mean(prevs, 0).view(-1, 1)
        aftr_res = torch.mean(aftrs, 0).view(-1, 1)

        out = self.referee(torch.cat((prev_res, aftr_res), 1).view(-1, 1, 2)
                           )[0][:, -1]
        out = self.final(out.t())

        return out.squeeze(-1)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
