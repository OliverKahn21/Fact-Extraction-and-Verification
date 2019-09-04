import os
import numpy as np
from matplotlib import pyplot as plt
from fever_io import load_dataset_json
from math import *
import re
import gensim.models.keyedvectors as word2vec


PATH = '/content/gdrive/My Drive/IRDM_CHFX2/'


def Subtask8_pre_1():
    """
    The output is a dictionary which the key is the document 'id' and the value is 'lines' in wiki-pages.
    """
    train_data = load_dataset_json(PATH + 'train.jsonl')

    evidence=[]
    for d in train_data.items():
        for i in range(5):
            evidence.append(d[1][i])

    files = os.listdir(PATH + 'data/wiki-pages/wiki-pages/')
    documents = {}
    for i in files:
        with open(os.path.join(PATH + 'data/wiki-pages/wiki-pages/', i)) as fp:
            lines = fp.readlines()
            for line in lines:
                line = eval(line)
                if line['id'] in evidence:
                    text = line['lines']
                    documents[line['id']] = text
    with open(PATH + 'Subtask8_pre_1.txt', 'w', encoding='utf-8') as f:
        f.write(str(documents))


Subtask8_pre_1()


def Subtask8_pre_2():
    """
    The output is two document. The 'pos.txt' including the claim and evidence sentence. The 'neg.txt' including the negative sample.
    """
    train_data = load_dataset_json(PATH + 'train.jsonl')

    with open(PATH + 'Subtask8_pre_1.txt', encoding='utf-8') as f:
        document = eval(f.read())

    pos = open(PATH + 'pos.txt', 'w', encoding='utf8')
    neg = open(PATH + 'neg.txt', 'w', encoding='utf8')
    for data in train_data:
        if data['label'] != 'NOT ENOUGH INFO':
            claim = data['claim'][:-1].lower()
            fp = pos if data['label'] == 'SUPPORTS' else neg
            fp.write(claim + '\n')
            for evidence in data['evidence']:
                # print(evidence)
                if evidence[0][2]:
                    tmp = evidence[0][2]
                    if tmp in document:
                        # print(document[tmp])
                        line = document[tmp].split('\n')[evidence[0][3]].replace(str(evidence[0][3]) + '\t', '')
                        # print(line)
                        fp.write(line + '\n')
            # break


Subtask8_pre_2()


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


def create_dictory():
    """
    This function is the summary of data pre-processing in Subtask2.
    """
    dictory = {}
    path = PATH + 'data/wiki-pages/wiki-pages/'
    files = os.listdir(path)
    D = 0
    documents = []
    for f in files:
        data = load_dataset_json(os.path.join(path, f))
        documents += data
        for d in data:
            D += 1
            text = list(set(d['text'].split(' ')))[1:]
            for t in text:
                t = re.sub("[,.。:_=+*&^%$#@!?()<>/`';|]", "", t)
                if t.isdigit():
                    continue
                if not t in dictory:
                    dictory[t] = [d['id']]
                else:
                    dictory[t].append(d['id'])
    print ('complete')
    return dictory, documents, D


def calculate_doc(claim_id, dictory, documents, D):
    """
    This function is the the optimized cosine similarly function in Subtask2
    The output of the function is dictionary which the key is the document id of the 5 most similar documents of the claim and the value is the document 'line' of each 'id'.
    """
    train_data = load_dataset_json(PATH + 'data/train.jsonl', instance_num=20)

    claim = None
    for d in train_data:
        if d['id'] == claim_id:
            d['claim'] = re.sub("[,.。:_=+*&^%$#@!?()<>/`';|]", "", d['claim'])
            d['claim'] = d['claim']
            claim = d['claim'].split(' ')
            break
    # print(d['id'] , d['claim'])

    keys = []
    for c in claim:
        tf = claim.count(c) / len(claim)
        idf = log((D / (1 + (len(dictory[c]) if c in dictory else 0))))
        keys.append((c, tf * idf, idf, tf, (len(dictory[c]) if c in dictory else 0)))
    keys.sort(key=lambda x: x[1], reverse=True)
    keys = keys[:5]
    vec1 = [k[1] for k in keys]
    # print(keys)

    document_tfidf = []
    for d in documents:
        text = d['text'].split(' ')
        vec2 = []
        for k in keys:
            tf = text.count(k[0]) / len(text)
            idf = k[2]
            vec2.append(tf * idf)
        sim = cosine_similarity(vec1, vec2)
        document_tfidf.append([d['id'], sim])
    document_tfidf.sort(key=lambda x: x[1], reverse=True)

    evidence = []
    for i in range(5):
        name = document_tfidf[i][0]
        evidence.append(name)

    files = os.listdir(PATH + 'data/wiki-pages/wiki-pages/')
    docu = {}
    for i in files:
        with open(os.path.join(PATH + 'data/wiki-pages/wiki-pages/', i)) as fp:
            lines = fp.readlines()
            for line in lines:
                line = eval(line)
                if line['id'] in evidence:
                    text = line['lines']
                    docu[line['id']] = text
    return docu


from cnn_pytorch import predict


def Subtak8_test():
    """
    The output is a list which including the top n test data.
    """
    test_data = []
    cnt = 0
    with open(PATH + 'data/test.jsonl') as fp:
        for line in fp.readlines():
            test_data.append(eval(line))
            cnt += 1
            if cnt == 10:
                break
    return test_data


dictory, documents, D = create_dictory()


def Subtask8():
    """
    Combine the function using in the previous task.
    Using a new convolutional neural networks for Sentence Classification.
    """
    with open(PATH + 'Q8_result.txt', 'w') as fp:
        test_data = Subtak8_test()
        for data in test_data:
            res = {}
            res['test_id'] = data['id']
            res['predicted_label'] = predict(data['claim'])
            res['predicted_evidence'] = []
            docs = calculate_doc(data['claim'], dictory, documents, D)
            claim = data['claim'].split(' ')
            keys = []
            for c in claim:
                tf = claim.count(c) / len(claim)
                idf = log((D / (1 + (len(dictory[c]) if c in dictory else 0))))
                keys.append((c, tf * idf, idf))
            keys.sort(key=lambda x: x[1], reverse=True)
            keys = keys[:5]
            vec1 = [k[1] for k in keys]

            for d in docs.items():
                lines = d[1].split('\n')
                sims = []
                for text in lines:
                    vec = []
                    for k in keys:
                        tf = text.count(k[0]) / len(text)
                        idf = k[2]
                        vec.append(tf * idf)
                    sim = cosine_similarity(vec1, vec)
                    sims.append(sim)
                # print(sims)
                index = sims.index(max(sims))
                res['predicted_evidence'].append((d[0], index))
            fp.write(str(res))


Subtask8()