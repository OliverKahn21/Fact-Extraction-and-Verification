import os
import numpy as np
from fever_io import load_dataset_json
from math import *
import re

PATH = '/content/gdrive/My Drive/IRDM_CHFX2/'


def n_dict_subtask3(filepath):
    """
    The input is the path of wiki-pages.
    The output is the dictionary which the key is the term and the value is the frequency of the term.
    The output is 98 MB so it is not including in the file.
    """
    n_dict = {}
    files = os.listdir(filepath)
    for i in files:
        with open(os.path.join(filepath, i)) as fp:
            lines = fp.readlines()
            for line in lines:
                text = eval(line)['text']  # extract data from the field of 'text'.
                words = text.split(' ')
                for w in words:
                    w = w.replace("-LRB-", "").replace("-RRB-", "").replace("-LSB-", "").replace("-RSB-", "").replace(
                        "--", "")
                    w = re.sub("[,.。:_=+*&^%$#@!?()<>/`';|]", "", w)  # replace the noisy with space.
                    if not w in n_dict:
                        n_dict[w] = 1
                    else:
                        n_dict[w] += 1  # count the frequencies of every term.
    np.save(PATH + "n_dict_Subtask3.npy", n_dict)
    print('save complete')


n_dict_subtask3(PATH + 'data/wiki-pages/wiki-pages/')


def Subtask3_0(claim_id):
    """
    The input is the claim id.
    The output is the claim, the 5 most similar documents and the query-likelihood unigram language model value.
    The out putis save in the 'Q3_unigram.csv'.
    In the query likelihood unigram language model, I do some smoothing to improve the result.
    For the terms in claim which do not appear in the document, the probability is not 0 but the probability it appear in the wiki-pages or p = 0.01 for the term even not appear in wiki-pages.
    """
    alpha = 0.5

    train_data = load_dataset_json(PATH + 'data/train.jsonl', instance_num=20)

    claim = None
    for d in train_data:
        if d['id'] == claim_id:
            claim = d['claim'][:-1]
            claim = re.sub("[,.。:_=+*&^%$#@!?()<>/`';|]", "", claim)
            claim = claim.split(' ')
            break
    print(d['id'])
    print(d['claim'])

    data = np.load(PATH + 'n_dict_Subtask3.npy', allow_pickle=True).item()
    C = sum(data.values())

    f = []
    files = os.listdir(PATH + 'data/wiki-pages/wiki-pages/')
    for i in files:
        with open(os.path.join(PATH + 'data/wiki-pages/wiki-pages/', i)) as fp:
            lines = fp.readlines()
            for line in lines:
                text = eval(line)['text']
                tmp = 0
                for w in claim:
                    if w in text:
                        p = text.count(w) / len(
                            text)  # calculate the probability for the terms appear in the document.
                    else:
                        if w in data:
                            p = alpha * data[
                                w] / C  # calculate the probability for the terms not appear in the document by using the probability it appear in the wiki-pages.
                        else:
                            p = 0.001  # the probability of the terms do not appear in wiki-pages is 0.001.
                    tmp += log(p)  # calculate the log(p) of the claim.
                f.append((eval(line)['id'], tmp))
    f.sort(key=lambda x: x[1], reverse=True)
    return f[:5]


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

for index in index_list:
    print(Subtask3_0(index))


def Subtask3_Laplace(claim_id):
    """
    The input is the claim id.
    The output is the claim, the 5 most similar documents and the Laplace Smoothing query-likelihood unigram language model value.
    The output is save in the 'Q3_laplace.csv'.
    """
    alpha = 0.5

    train_data = load_dataset_json(PATH + 'data/train.jsonl', instance_num=20)

    claim = None
    for d in train_data:
        if d['id'] == claim_id:
            claim = d['claim'][:-1]
            claim = re.sub("[,.。:_=+*&^%$#@!?()<>/`';|]", "", claim)
            claim = claim.split(' ')
            break
    print(d['id'])
    print(d['claim'])

    data = np.load(PATH + 'n_dict_Subtask3.npy', allow_pickle=True).item()
    C = sum(data.values())
    f = []

    files = os.listdir(PATH + 'data/wiki-pages/wiki-pages/')
    for i in files:
        with open(os.path.join(PATH + 'data/wiki-pages/wiki-pages/', i)) as fp:
            lines = fp.readlines()
            for line in lines:
                text = eval(line)['text'].split(' ')
                tmp = 0
                for w in claim:
                    if w in text:
                        p = (text.count(w) + 1) / (
                                    len(text) + 1)  # calculate the probability for the terms appear in the document.
                    else:  # calculate the probability for the terms not appear in the document by using the Laplace.
                        if w in data:
                            p = alpha * (data[w] + 1) / (C + len(
                                data))
                        else:
                            p = 0.01  # the probability of the terms do not appear in wiki-pages is 0.001.
                    tmp += log(p)  # calculate the log(p) of the claim.
                f.append((eval(line)['id'], tmp))
    f.sort(key=lambda x: x[1], reverse=True)
    return f[:5]


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

for index in index_list:
    print(Subtask3_Laplace(index))


def Subtask3_JM(claim_id):
    """
    The input is the claim id.
    The output is the claim, the 5 most similar documents and the Jelinek-Mercer Smoothing query-likelihood unigram language model value.
    The output is save in the 'Q3_jelinek_mercer.csv'.
    """
    alpha = 0.5

    train_data = load_dataset_json(PATH + 'data/train.jsonl', instance_num=20)

    claim = None
    for d in train_data:
        if d['id'] == claim_id:
            claim = d['claim'][:-1]
            claim = re.sub("[,.。:_=+*&^%$#@!?()<>/`';|]", "", claim)
            claim = claim.split(' ')
            break
    print(d['id'])
    print(d['claim'])

    data = np.load(PATH + 'n_dict_Subtask3.npy', allow_pickle=True).item()
    C = sum(data.values())
    f = []

    files = os.listdir(PATH + 'data/wiki-pages/wiki-pages/')
    for i in files:
        with open(os.path.join(PATH + 'data/wiki-pages/wiki-pages/', i)) as fp:
            lines = fp.readlines()
            for line in lines:
                text = eval(line)['text'].split(' ')
                tmp = 0
                for w in claim:
                    if w in data:
                        p = alpha * text.count(w) / len(text) + (1 - alpha) * data[
                            w] / C  # calculate the probability for the terms appear in the document by using the JK.
                    else:
                        p = alpha * text.count(w) / len(
                            text)  # calculate the probability for the terms not appear in the document by using the JK.
                    tmp += log(p)  # calculate the log(p) of the claim.
                f.append((eval(line)['id'], tmp))
    f.sort(key=lambda x: x[1], reverse=True)
    return f[:5]


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

for index in index_list:
    print(Subtask3_JM(index))


def Subtask3_Dirichlet(claim_id):
    """
    The input is the claim id.
    The output is the claim, the 5 most similar documents and the Dirichlet Smoothing query-likelihood unigram language
    model value.
    The output is save in the 'Q3_dirichlet.csv'.
    """
    train_data = load_dataset_json(PATH + 'data/train.jsonl', instance_num=20)

    claim = None
    for d in train_data:
        if d['id'] == claim_id:
            claim = d['claim'][:-1]
            claim = re.sub("[,.。:_=+*&^%$#@!?()<>/`';|]", "", claim)
            claim = claim.split(' ')
            break
    print(d['id'])
    print(d['claim'])

    data = np.load(PATH + 'n_dict_Subtask3.npy', allow_pickle=True).item()
    C = sum(data.values())
    N = 5396106
    u = C / N
    f = []

    files = os.listdir(PATH + 'data/wiki-pages/wiki-pages/')
    for i in files:
        with open(os.path.join(PATH + 'data/wiki-pages/wiki-pages/', i)) as fp:
            lines = fp.readlines()
            for line in lines:
                text = eval(line)['text'].split(' ')
                alpha = u / (len(text) + u)
                tmp = 0
                for w in claim:
                    if w in data:
                        p = (text.count(w) + u * data[w] / C) / (len(
                            text) + u)  # calculate the probability for the terms appear in document with Dirichlet.
                    else:
                        p = (text.count(w)) / (len(
                            text) + u)  # calculate the probability for the terms not appear in document with Dirichlet.
                    tmp += log(p)  # calculate the log(p) of the claim.
                f.append((eval(line)['id'], tmp))
    f.sort(key=lambda x: x[1], reverse=True)
    return f[:5]


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

for index in index_list:
    print(Subtask3_Dirichlet(index))
