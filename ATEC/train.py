# coding:utf-8


from dataprocess import Pair,csv_file, word_dic_file, corpus_save_file
import utils.textProcess as tp
from NeuralNetworkUtils.utils.utils import split2Batches
from model import embedding_sum_model, embedding_sum_model_square_distance
import tensorflow as tf
import numpy as np
tensorboard_path='./data/tensorboard'
model_save_file='./data/mode_save/sum_word_embedding'


# corpus = tp.loadPickle(corpus_save_file)
# #
# train_X = []
# train_Y = []


def build_train_set():
    sen_len = 25
    count = 0
    train_X = []
    train_Y = []
    for pair in corpus:
        sample = [pair.padding_index_first_sen, pair.padding_index_second_sen]
        train_X.append(sample)
        train_Y.append(int(pair.label))
        count += 1
    return train_X, train_Y


def statistic_correct(pred, label):
    pred = [np.argmax(i) for i in pred]
    label = [np.argmax(i) for i in label]
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    # int_pred = [int(p + 0.5) for p in pred]
    for i in range(len(pred)):
        if pred[i] == label[i]:
            if label[i] == 0:
                TN += 1
            if label[i] == 1.0:
                TP += 1
        else:
            if label[i] == 0:
                FP += 1
            if label[i] == 1.0:
                FN += 1
    return TP, TN, FP, FN

def measure(TP, TN, FP, FN):
    if TP+TN+FP+FN==0: acc=0
    else: acc = 1.0 * (TP + TN) / (TP+TN+FP+FN)
    if TP + FP==0: precision =0
    else: precision = 1.0 * TP / (TP + FP)
    if TP + FN==0:recall=0
    else:recall = 1.0 * TP / (TP + FN)
    if precision+recall==0: f1=0
    else: f1 = 2 * precision * recall / (precision + recall)
    return acc,precision,recall,f1

index_dic = tp.loadPickle(word_dic_file)
# train_X, train_Y = build_train_set()
# # # train_Y = [[i] for i in train_Y]
# train_Y = tp.oneHotLabels(train_Y, 1)
# train_X, train_Y, batch_num = split2Batches(20, train_X, train_Y)
# train_X = [train_X[0]]
# train_Y = [train_Y[0]]
# tp.savePickle(train_X,'./tmpX')
# tp.savePickle(train_Y,'./tmpY')
X = tp.loadPickle('./tmpX')
Y = tp.loadPickle('./tmpY')

train_X = X[0:-100]
train_Y = Y[0:-100]
test_X = []
for i in X[-100:-2]:
    test_X.extend(i)
test_Y = []
for i in Y[-100:-2]:
    test_Y.extend(i)
print 'finish loading'
batch_num=len(train_X)
with tf.Session() as sess:
    net = embedding_sum_model_square_distance(sen_dim=25, vocab_dim=len(index_dic) + 2, word_dim=50)


    sess.run(tf.global_variables_initializer())
    summary = tf.summary.FileWriter(tensorboard_path)
    summary.add_graph(sess.graph, global_step=1)
    obop = tf.summary.merge_all()
    for epoch in range(50):
        sta = [0] * 5
        for i in range(batch_num-20):
            results = sess.run([net['loss'], net['opt'], net['pred']],
                               feed_dict={net['x']: train_X[i], net['y']: train_Y[i]})
            sta[0] += results[0]
            mea = statistic_correct(results[2], train_Y[i])
            for cnt in range(1, 5):
                sta[cnt] += mea[cnt - 1]
        ops = sess.run(obop)
        summary.add_summary(ops, epoch)
        acc,precision,recall, f1 =measure(sta[1],sta[2],sta[3],sta[4])
        print 'epoch-> ',epoch,'loss:', 1.0 * sta[0] / batch_num
        print '\tacc:', acc,
        print '\tprecision:', precision,
        print '\trecall:', recall,
        print '\tf1:', f1
    saver = tf.train.Saver()
    saver.save(sess, model_save_file)

def predict():

    results = sess.run([net['pred']],
                               feed_dict={net['x']: test_X})
    a,b,c,d = statistic_correct(results[0], test_Y)
    acc, precision, recall, f1 = measure(a,b,c,d)

    print '\tacc:', acc,
    print '\tprecision:', precision,
    print '\trecall:', recall,
    print '\tf1:', f1


