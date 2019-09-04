import os
import re
import numpy as np
from matplotlib import pyplot as plt
from math import *
import pandas as pd

PATH = '/content/gdrive/My Drive/IRDM_CHFX2/'


def n_dict_subtask1(filepath):
    """
    The input is the path of wiki-pages.
    The output is a dictionary which the key is the terms and the value is the freuquency.
    The output is 90MB so it is not including in the file.
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
                    w = w.replace("-LRB-", "").replace("-RRB-", "").replace("-LSB-", "") \
                        .replace("-RSB-", "").replace("--", "")
                    w = w.lower()
                    w = re.sub("[,.。:_=+*&^%$#@!?()<>/`';|]", "", w)  # replace the noisy with space.
                    if w not in n_dict:
                        n_dict[w] = 1
                    else:
                        n_dict[w] += 1  # count the frequencies of every term.
    np.save(PATH + "n_dict_Subtask1.npy", n_dict)
    print('save complete')


n_dict_subtask1(PATH + 'data/wiki-pages/wiki-pages/')


def Subtask1(n_dict_npy):
    """
    The input is the path of n_dict_Subtask1.npy.
    The output is the Zipf's Law graph and the CSV of all the terms and there frequencies.
    The output is 300MB so it is not including in the file. I extract the result of top 500
    terms and save it in the file as 'Q1_term_frequency.csv'.
    """
    n_dict = np.load(n_dict_npy, allow_pickle=True).item()
    data = sorted(n_dict.items(), key=lambda item: n_dict[item[0]], reverse=True)  # sort the data.
    total = 0
    for d in data:
        total += d[1]  # count the total number of words.
    df = pd.DataFrame(data, columns=['term', 'Freq'])[1:]  # exclude 'space'
    df['rank'] = df.index
    df['Pr(%)'] = df['Freq'].div(total) * 100
    df['r*Pr'] = df['rank'] * df['Pr(%)'] / 100
    df_plot = df[:100]  # plot the top 100 terms.
    # data = data[:100]
    plt.plot(df_plot['rank'], df_plot['Pr(%)'], 'b-')
    plt.plot(np.arange(0, 100, 1), 10 / np.arange(0, 100, 1), 'r:')
    # plt.plot([d[0] for d in data], [d[1] / total for d in data])
    plt.yticks(np.arange(0, 10, 0.5))
    plt.xticks(np.arange(0, 101, 10))
    plt.ylabel('Probability(%)')
    plt.xlabel('Rank (by decreasing frequency)')
    plt.title("Zipf's Law")
    plt.show()
    plt.savefig('res.jpg')
    print(total)
    print(df.head(25))
    df.to_csv(PATH + 'Subtask1.csv')


Subtask1(PATH + 'n_dict_Subtask1.npy')
