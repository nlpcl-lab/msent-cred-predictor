# Src

The main folder, includes source codes written in Python 3.

## parser.py

This file extracts some information we need from the full data.
Parsed things are saved in the `../data/docs` folder.

Options:

- `--data`: a path to the data file. Default: `../data/credon.json`, str

Example:L

```bash
python parser.py --data=../data/credon.json
```

## train.py

The main file trains the model by the training set, also tests the accuracy with the test set.

options:

- `--docs`: A path to the docs made by `parser.py` above. Default: `../data/docs`, str
- `--epoch`: A number of train epoch. Default: `5`, int
- `--lr`: Learning rate. Default: `2e-5`, float
- `--bert_type`: A name of the bert model we use for sentence embedding. Default: `bert-base-multilingual-cased`, str
- `--cuda`: A T/F parameter for using CUDA(GPU). Default: `False`, bool
- `--log`: A T/F parameter for logging for _tensorboard_. Default: `False`, bool

## models.py

Contains input class & pytorch models. The main model uses BERT from pytorch_transformers, and two LSTMCells.

## utils.py

Contains customized, usefull utils for tokenizing & debugging.
