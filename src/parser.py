import os
import json
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data/credon.json')
args = parser.parse_args()

CRED_IDX1 = 'Perceived_Author_Credibility_for_the_upcoming_sentences'
CRED_IDX2 = 'Perceived_Author_Credibility'

CRED_DICT = {
    ('_').join('Strong Credibility for the upcoming sentences'.split()): 7.0,
    ('_').join('Credibility for the upcoming sentences'.split()): 6.0,
    ('_').join('Weak Credibility for the upcoming sentences'.split()): 5.0,
    ('_').join('Hard to Judge'.split()): 4.0,
    ('_').join('Weak Suspicion for the upcoming sentencese'.split()): 3.0,
    ('_').join('Suspicion for the upcoming sentences'.split()): 2.0,
    ('_').join('Strong Suspicion for the upcoming sentences'.split()): 1.0
}

valid_docs = list()


def get_score(body):
    if CRED_IDX1 in body:
        return CRED_DICT[body[CRED_IDX1]['value']]
    return CRED_DICT[body[CRED_IDX2]['initial_value']]


class Sentence:
    def __init__(self, sdata):
        self.idx = sdata['identifier']['sentence_index']
        self.body = sdata['body']['sentence'].strip()
        self.ann_ids = sdata['down'].get('sentence')
        self.anns = list()

    def load_annotations(self, fulldata):
        for aid in self.ann_ids:
            said = fulldata[aid]['down']['sentence_annotation'][0]
            self.anns.append(get_score(fulldata[said]['body']))

    def _avg(self):
        return (sum(self.anns) / len(self.anns)) if self.anns else 0.0

    def is_valid(self):
        return self._avg() == 0.0

    def __repr__(self):
        return '%d\t%s\t%f' % (self.idx, self.body, self._avg())


class Document:
    def __init__(self, docdata):
        self.id = int(docdata['identifier']['doc_id'])
        self.sentence_ids = docdata['down']['sentence']
        self.sentences = list()
        self.valid = False

    def load_sentences(self, fulldata):
        for sid in self.sentence_ids:
            self.sentences.append(Sentence(fulldata[sid]))
        self.sentences.sort(key=lambda x: x.idx)

        with open('../data/docs/%s.tsv' % self.id, 'w', encoding='UTF-8') as f:
            tmp_sum = 0.0
            for sentence in self.sentences:
                if sentence.ann_ids:
                    sentence.load_annotations(fulldata)

                tmp_sum += 1 if sentence.is_valid() else 0
                f.write(str(sentence))
                f.write('\n')

            if tmp_sum < 10:
                global valid_docs
                valid_docs.append(self.id)

    def length(self):
        return len(self.sentences)

    def __repr__(self):
        return 'Document class: id %d' % self.id


def get_full_data(path):
    f = open(path, 'r', encoding='UTF-8')
    jsondat = json.load(f)
    f.close()

    return jsondat[0]


def get_docs(fulldata):
    docs = list(filter(lambda x: fulldata[x]['type'] == 'document', fulldata))
    docs = list(map(lambda x: fulldata[x], docs))
    docs = list(map(Document, docs))
    docs.sort(key=lambda x: x.id)

    print('Number of docs: %d' % len(docs))
    return docs


if __name__ == '__main__':
    fulldata = get_full_data(args.data)
    docs = get_docs(fulldata)

    for doc in tqdm.tqdm(docs):
        doc.load_sentences(fulldata)

    print('Number of valid documents: %d' % len(valid_docs))

    for doc in tqdm.tqdm(docs):
        if not doc.id in valid_docs:
            os.remove('../data/docs/%d.tsv' % doc.id)
