# Multi-Sentence Credibility Predictor

A credibility predictor using relations with other sentences and credibility scores.  
The model mainly use [pytorch version](https://github.com/huggingface/pytorch-transformers) of [_BERT_](https://github.com/google-research/bert), and two LSTMCells as its structure.

## Setup

Requires **Python 3.6** or higher.
Recommended to use virtualenv or anaconda.

```bash
pip install torch
pip install tqdm
pip install spacy
python -m spacy download en
pip install tensorboardX
pip install pytorch_transformers
```

It is not easy to provide the requirements in pip-installable file format (like `requirements.txt`) because of the module _spaCy_.

After installation, place a json file in `data` folder which contains full data of credon.

## Description for each folder

- [`cache`](./cache/)
- [`data`](./data/)
- [`src`](./src/)

## Usage example

```bash
cd src
python parser.py --data=../data/credon.json
python train.py --docs=../data/docs --epoch=4 --cuda --log
```

## Contributors

- [Junseop](https://github.com/gaonnr)
