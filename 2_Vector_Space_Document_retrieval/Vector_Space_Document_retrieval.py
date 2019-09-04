import numpy as np
from fever_io import load_dataset_json
from math import *
import re
import json

PATH = '/content/gdrive/My Drive/IRDM_CHFX2/'


def cosine_similarity(x, y, norm=False):
    """
    The input x is the TF-IDF of the claim.
    The input y is the TF-IDF of each documents.
    The output is the cosine similarity with normalization between 0 and 1.
    """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos


def extract_wiki(wikipedia_dir):
    """
    The input is the path of wiki-pages.
    The output is a dictionary which the key is the 'id' and the value is the 'text' which 'text' is not empty.
    The output is 3 GB so it is not including in the file.
    """
    diction = dict()
    for i in range(1, 110):  # jsonl file number from 001 to 109
        jnum = "{:03d}".format(i)
        fname = wikipedia_dir + "wiki-" + jnum + ".jsonl"
        with open(fname) as f:
            line = f.readline()
            while line:
                data = json.loads(line.rstrip("\n"))
                doc_id = data["id"]
                text = data["text"]
                if text != "":
                    diction[doc_id] = text
                line = f.readline()
    np.save(PATH + "diction_Subtask2.npy", diction)
    print("save complete")


extract_wiki(PATH + "data/wiki-pages/wiki-pages/")


def create_dictory(diction):
    """
    The purpose of this function is inverted index
    The input is the path of 'diction_Subtask2.npy'
    The output is a dictionary which the key is the term and the value is the id of the documents including this term.
    It also print the total number of documents.
    The output is 2 GB so it is not including in the file.
    """
    data = np.load(diction, allow_pickle=True).item()
    dictory = {}
    n_ducuments = len(data)
    for d in data.items():
        text = list(set(d[1].split(' ')))[1:]
        for t in text:
            t = re.sub("[,.。:_=+*&^%$#@!?()<>/`';|]", "", t)
            if t.isdigit():
                continue
            if not t in dictory:
                dictory[t] = [d[0]]
            else:
                dictory[t].append(d[0])
    np.save(PATH + "dictory_Subtask2.npy", dictory)
    print(n_ducuments)


create_dictory(PATH + 'diction_Subtask2.npy')


def Subtask2_cossim(claim_id, numberofducuments):
    """
    The input is the list of claim 'id' and the total number of documents.
    The output is the claim, the top 5 TF-IDF terms in the claim, the TF-IDF of these terms, the five most similar documents with the claim and the cosine similarity between them.
    The top 5 TF-IDF terms in the claim and the TF-IDF of these terms is save in the 'Q2_claim_TF-IDF.csv' and the claim with the five most similar documents with the claim is save in the 'Q2_vector_space.csv'.
    """
    data = np.load(PATH + 'diction_Subtask2.npy', allow_pickle=True).item()
    dictory = np.load(PATH + 'dictory_Subtask2.npy', allow_pickle=True).item()
    train_data = load_dataset_json(PATH + 'data/train.jsonl', instance_num=20)

    claim = None
    for d in train_data:
        if d['id'] == claim_id:
            d['claim'] = re.sub("[,.。:_=+*&^%$#@!?()<>/`';|]", "", d['claim'])
            claim = d['claim'].split(' ')
            break
    print(d['id'])
    print(d['claim'])

    keys = []
    for c in claim:
        tf = claim.count(c) / len(claim)
        idf = log((numberofducuments / (1 + (len(dictory[c]) if c in dictory else 0))))
        keys.append((c, tf * idf, idf, tf))
    keys.sort(key=lambda x: x[1], reverse=True)
    keys = keys[:5]
    word = [k[0] for k in keys]
    vec1 = [k[1] for k in keys]  # vec1 is the list of tf*IDF of top 5 words in the claim.
    print(word)
    print(vec1)

    document_tfidf = []
    for d in data.items():
        text = d[1].split(' ')
        vec2 = []
        for k in keys:
            tf = text.count(k[0]) / len(text)
            idf = k[2]
            vec2.append(tf * idf)  # vec2 is the list of tf*IDF of top 5 words in the document.
        sim = cosine_similarity(vec1, vec2)
        document_tfidf.append([d[0], sim])
    document_tfidf.sort(key=lambda x: x[1], reverse=True)
    return document_tfidf[:5]


index_list = [
    75397,
    150448,
    214861,
    156709,
    129629,
    33078,
    6744,
    226034,
    40190,
    76253]
numberofducuments = 5396106

for index in index_list:
    print(Subtask2_cossim(index, numberofducuments))
