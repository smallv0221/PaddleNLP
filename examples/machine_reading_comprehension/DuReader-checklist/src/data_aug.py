import math
from six import iteritems
from six.moves import xrange
import jieba.posseg as pseg
import codecs
from gensim import corpora
import json
import os
import re

from paddlenlp.datasets import load_dataset

# BM25 parameters.
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(map(lambda x: float(len(x)),
                             corpus)) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.initialize()

    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(
                freq + 0.5)

    def get_score(self, document, index, average_idf):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[
                word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1) /
                      (self.f[index][word] + PARAM_K1 *
                       (1 - PARAM_B + PARAM_B * self.corpus_size / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores


def get_bm25_weights(corpus):
    bm25 = BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(
        bm25.idf.keys())

    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)

    return weights


stop_words = 'baidu_stopwords.txt'
stopwords = codecs.open(stop_words, 'r', encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]

stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']


def tokenization(context):
    result = []
    words = pseg.cut(context)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result


train_ds, dev_ds = load_dataset('dureader_robust', splits=['train', 'dev'])
corpus = []
for i in range(len(train_ds)):
    context = train_ds[i]['context']
    corpus.append(tokenization(context))

dictionary = corpora.Dictionary(corpus)

bm25Model = BM25(corpus)
average_idf = sum(
    map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(
        bm25Model.idf.keys())

with open('train.json', "r", encoding="utf8") as f:
    input_data = json.load(f)

for i in range(len(train_ds)):
    query_str = train_ds[i]['question']
    query = []
    for word, flag in pseg.cut(train_ds[i]['question']):
        query.append(word)
    scores = bm25Model.get_scores(query, average_idf)
    scores[i] = 0
    idx = scores.index(max(scores))
    example = {
        'context': train_ds[idx]['context'],
        "qas": [{
            "question": train_ds[i]['question'],
            "id": str(i) + '-' + str(idx),
            "answers": []
        }]
    }

    input_data['data'][0]['paragraphs'].append(example)

with open('train_aug.json', "w", encoding="utf8") as f:
    f.write(json.dumps(input_data, indent=4, ensure_ascii=False) + "\n")
