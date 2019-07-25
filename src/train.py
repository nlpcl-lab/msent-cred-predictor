import tqdm
import torch
import argparse
import numpy as np

from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import model

from utils import get_docnames
from utils import pretty_print
from utils import str2bool

from esim.model import ESIM

parser = argparse.ArgumentParser()
parser.add_argument('--docs', type=str, default='../data/docs')
parser.add_argument('--epoch', type=int, default=256)
parser.add_argument('--lr', type=float, default=6e-4)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--cuda', type=str2bool, default='True')
parser.add_argument('--esim', type=str, default='../data/best.pth.tar')


def load_esim(chkp_path):
    chkp = torch.load(chkp_path)

    vocab_size = chkp['model']['_word_embedding.weight'].size(0)
    embd_dim = chkp['model']['_word_embeding.weight'].size(1)
    hddn_size = chkp['model']['_projection.0.weight'].size(0)
    num_classes = chkp['model']['_classification.4.weight'].size(0)

    esim = ESIM(vocab_size, embd_dim, hddn_size, num_classes=num_classes)
    esim.load_state_dict(chkp['model'])

    return esim


if __name__ == '__main__':
    args = parser.parse_args()
    dlist = get_docnames(args.docs)
    docs = [model.Document(x) for x in dlist]

    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    esim = load_esim(args.esim)
