import os
import numpy as np
from matplotlib import pyplot as plt
from fever_io import load_dataset_json
from math import *
import re
import gensim.models.keyedvectors as word2vec


PATH = '/content/gdrive/My Drive/IRDM_CHFX2/'


def prepare_train_Subtask6_1():
    """
    This function is the same as the function uss in the Subtask4 and the purpose of it is to get a dictionary which the key is document id and the value is the 'line'.
    The instance number we use at this project is 5000.
    """
    train_data = load_dataset_json(PATH + 'data/train.jsonl', instance_num = 5000)
    evidence = []
    for i in train_data:
        for j in i['evidence']:
            evidence.append(j[0][2])
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
    with open(PATH + 'prepare_train_Subtask6_1.txt', 'w', encoding='utf-8') as f:
        f.write(str(documents))


prepare_train_Subtask6_1()


def prepare_train_Subtask6_2():
    """
    This function aim to connect the claim, the evidence sentence and the label. And embedding this list.
    The output of this function is the train data for the neural network and it is a 601-dimensional vector for training
    """
    with open(PATH + 'prepare_train_Subtask6_1.txt', encoding='utf-8') as f:
        document = eval(f.read())

    train_data = load_dataset_json(PATH + 'data/train.jsonl', instance_num=5000)
    model = word2vec.KeyedVectors.load_word2vec_format(PATH + "data/GoogleNews-vectors-negative300.bin", binary=True)

    with open(PATH + 'traindata_Subtask6.txt', 'w') as fp:
        for data in train_data:
            if data['label'] != 'NOT ENOUGH INFO':
                claim = data['claim'][:-1]
                claim = re.sub("[-,.。:_=+*&^%$#@!?()<>/`';|]", "", claim)
                claim = claim.split(' ')
                claim = list(filter(lambda x: x in model.vocab, claim))
                Vi = []
                for i in range(len(claim)):
                    Vi.append(model[claim[i]])

                V = np.zeros(len(Vi[0]))
                for i in range(len(claim)):
                    for j in range(len(Vi[0])):
                        V[j] = V[j] + Vi[i][j]

                rms = 0
                for i in range(len(Vi[0])):
                    rms += V[i] * V[i]
                rms = np.sqrt(rms / len(Vi[0]))

                for i in range(len(Vi[0])):
                    V[i] = V[i] / rms

                label = '1' if data['label'] == 'SUPPORTS' else '0'
                # V = V.astype(str).tolist()

                for evidence in data['evidence']:
                    # print(evidence)
                    if evidence[0][2]:
                        tmp = evidence[0][2]
                        if tmp in document:
                            # print(document[tmp])
                            lines = document[tmp].split('\n')
                            # for k in range(len(lines)):
                            line = document[tmp].split('\n')[evidence[0][3]].replace(str(evidence[0][3]) + '\t', '')
                            line = re.sub('[-,.。:_=+*&^%$#@!?()<>/]', '', line)
                            line = line.split(' ')
                            line = list(filter(lambda x: x in model.vocab, line))
                            # print(line)
                            Vi = []
                            for i in range(len(line)):
                                Vi.append(model[line[i]])
                            V1 = np.zeros(len(Vi[0]))
                            for i in range(len(line)):
                                for j in range(len(Vi[0])):
                                    V1[j] = V1[j] + Vi[i][j]
                            rms = 0
                            for i in range(len(Vi[0])):
                                rms += V1[i] * V1[i]
                            rms = np.sqrt(rms / len(Vi[0]))
                            for i in range(len(Vi[0])):
                                V1[i] = V1[i] / rms
                            # res = V - V1
                            # print(type(V))
                            res1 = V.astype(str).tolist()
                            res2 = V1.astype(str).tolist()

                            fp.write(' '.join(res1) + ' ' + ' '.join(res2) + ' ' + label + '\n')
                # break


prepare_train_Subtask6_2()


'''
Following two functions is the same as the above two function. The only different is following two functions output the test data from dev data set.
'''


def prepare_dev_Subtask6_1():
    train_data = load_dataset_json(PATH + 'data/dev.jsonl', instance_num=500)
    evidence = []
    for i in train_data:
        for j in i['evidence']:
            evidence.append(j[0][2])
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
    with open(PATH + 'prepare_dev_Subtask6_1.txt', 'w', encoding='utf-8') as f:
        f.write(str(documents))


prepare_dev_Subtask6_1()


def prepare_dev_Subtask6_2():
    with open(PATH + 'prepare_dev_Subtask6_1.txt', encoding='utf-8') as f:
        document = eval(f.read())

    train_data = load_dataset_json(PATH + 'data/dev.jsonl', instance_num=500)
    model = word2vec.KeyedVectors.load_word2vec_format(PATH + "data/GoogleNews-vectors-negative300.bin", binary=True)

    with open(PATH + 'devdata_Subtask6.txt', 'w') as fp:
        for data in train_data:
            if data['label'] != 'NOT ENOUGH INFO':
                claim = data['claim'][:-1]
                claim = re.sub("[-,.。:_=+*&^%$#@!?()<>/`';|]", "", claim)
                claim = claim.split(' ')
                claim = list(filter(lambda x: x in model.vocab, claim))
                Vi = []
                for i in range(len(claim)):
                    Vi.append(model[claim[i]])

                V = np.zeros(len(Vi[0]))
                for i in range(len(claim)):
                    for j in range(len(Vi[0])):
                        V[j] = V[j] + Vi[i][j]

                rms = 0
                for i in range(len(Vi[0])):
                    rms += V[i] * V[i]
                rms = np.sqrt(rms / len(Vi[0]))

                for i in range(len(Vi[0])):
                    V[i] = V[i] / rms

                label = '1' if data['label'] == 'SUPPORTS' else '0'
                # V = V.astype(str).tolist()

                for evidence in data['evidence']:
                    # print(evidence)
                    if evidence[0][2]:
                        tmp = evidence[0][2]
                        if tmp in document:
                            # print(document[tmp])
                            lines = document[tmp].split('\n')
                            # for k in range(len(lines)):
                            line = document[tmp].split('\n')[evidence[0][3]].replace(str(evidence[0][3]) + '\t', '')
                            line = re.sub('[-,.。:_=+*&^%$#@!?()<>/]', '', line)
                            line = line.split(' ')
                            line = list(filter(lambda x: x in model.vocab, line))
                            # print(line)
                            Vi = []
                            for i in range(len(line)):
                                Vi.append(model[line[i]])
                            V1 = np.zeros(len(Vi[0]))
                            for i in range(len(line)):
                                for j in range(len(Vi[0])):
                                    V1[j] = V1[j] + Vi[i][j]
                            rms = 0
                            for i in range(len(Vi[0])):
                                rms += V1[i] * V1[i]
                            rms = np.sqrt(rms / len(Vi[0]))
                            for i in range(len(Vi[0])):
                                V1[i] = V1[i] / rms
                            # res = V - V1
                            # print(type(V))
                            res1 = V.astype(str).tolist()
                            res2 = V1.astype(str).tolist()

                            fp.write(' '.join(res1) + ' ' + ' '.join(res2) + ' ' + label + '\n')
                # break


prepare_dev_Subtask6_2()


import pandas as pd
import tensorflow as tf


'''
Translating the train data and test data to the format that tensorflow can read.
'''
COLUMN_NAMES = []
for i in range(600):
    COLUMN_NAMES.append('a' + str(i) )
COLUMN_NAMES.append('label')
data_train = pd.read_csv(PATH + 'traindata_Subtask6.txt',sep=' ', names=COLUMN_NAMES, header=0)
data_test = pd.read_csv(PATH + 'devdata_Subtask6.txt',sep=' ', names=COLUMN_NAMES, header=0)
train_x, train_y = data_train, data_train.pop('label')
test_x, test_y = data_test, data_test.pop('label')


'''
Extracting the feature of data from train set.
'''
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)


def my_model_fn(features, labels, mode, params):
    '''
    The purpose of this function is to define my custom estimator.
    '''
    net = tf.feature_column.input_layer(features,
                                        params['feature_columns'])  # define the input layer by the feature columns.

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units,
                              activation=tf.nn.relu)
        # define the hidden layer by the number of hidden layer and the number of neurons in each layer

    logits = tf.layers.dense(net, params['n_classes'],
                             activation=None)  # define the output layer by the classify number.

    # define the prediction part
    predicted_classes = tf.argmax(logits, 1)  # The maximum result is the prediction result.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],  # change it to the list [[a],[b]]
            'probabilities': tf.nn.softmax(logits),  # normalization
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # define the loss function.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # define the training process.
    if mode == tf.estimator.ModeKeys.TRAIN:  # optimize the loss function to reduce the loss and improve the accuracy.
        # optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        # optimizer = tf.train.MomentumOptimizer(learning_rate = 0.001)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())  # optimization
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # define the evaluation process.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes)
    auc = tf.metrics.auc(labels=labels,
                         predictions=predicted_classes)
    precision = tf.metrics.precision(labels=labels,
                                     predictions=predicted_classes)
    recall = tf.metrics.recall(labels=labels,
                               predictions=predicted_classes)
    metrics = {'accuracy': accuracy, 'AUC': auc, 'precision': precision, 'recall': recall}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


'''
Define the model, including the model, hidden layer and the number of classification
'''
classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir=PATH + 'model_Subtask6/',
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': [512, 512],
            'n_classes': 2,
        })


'''
The number of training examples utilized in one iteration is 256 and randomly adjust the data order in each iteration.
'''
batch_size = 128


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)  # randomly adjust the data order
    return dataset.make_one_shot_iterator().get_next()


'''
Begin training and the step is 2500.
'''
classifier.train(
    input_fn=lambda:train_input_fn(train_x, train_y, batch_size),
    steps = 2500)


'''
Define the testing process.
'''


def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    inputs=(features,labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


'''
Evaluate our model.
'''
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y,batch_size))

print(eval_result)

