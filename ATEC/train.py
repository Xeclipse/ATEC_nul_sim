# coding:utf-8


from dataprocess import Pair
import utils.textProcess as tp
from NeuralNetworkUtils.utils.utils import split2Batches
from model import embedding_sum_model, embedding_sum_model_square_distance
import tensorflow as tf
import numpy as np
# #
# corpus = tp.loadPickle('/Users/nali/PycharmProjects/MachineLearningLaboratory/ATEC/data/processed_data/corpus')
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
    acc = 1.0 * (TP + TN) / (TP+TN+FP+FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return acc,precision,recall,f1

index_dic = tp.loadPickle('/Users/nali/PycharmProjects/MachineLearningLaboratory/ATEC/data/processed_data/word_dic.dic')
# train_X, train_Y = build_train_set()
# # # train_Y = [[i] for i in train_Y]
# train_Y = tp.oneHotLabels(train_Y, 1)
# train_X, train_Y, batch_num = split2Batches(20, train_X, train_Y)
# # train_X = [train_X[0]]
# # train_Y = [train_Y[0]]
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
    # 'x':x,
    # 'y':y,
    # 'loss':loss,
    # 'pred':pred,
    # 'opt':optimizer
    net = embedding_sum_model_square_distance(sen_dim=25, vocab_dim=len(index_dic) + 2, word_dim=100)


    # sen_dim=25
    # vocab_dim= len(index_dic)+2
    # word_dim = 10
    #
    # x = tf.placeholder(dtype=tf.int32, shape=[None, 2, sen_dim], name="input")
    # y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="label")
    #
    # intialembedding = tf.random_normal(shape=[vocab_dim, word_dim], mean=1.0, stddev=1.0)
    # word_vector_table = tf.Variable(intialembedding, name='word_vector')
    # tf.summary.histogram(name='embedding', values=word_vector_table)
    #
    # # all_embedding -> [batch, 2, sen_dim, word_dim]
    # all_embedding = tf.nn.embedding_lookup(word_vector_table, x)
    # # sum_embedding ->[batch, 2, word_dim]
    # sum_embedding = tf.reduce_sum(all_embedding, axis=2)
    # # normalize_embedding->[batch, 2, word_dim]
    # normalize_embedding = tf.nn.l2_normalize(sum_embedding, dim=1)
    # # calc cosine distance
    # trans_embedding = tf.transpose(normalize_embedding, perm=[1, 0, 2])
    # left_embedding = tf.slice(trans_embedding, [0, 0, 0], [1, -1, -1])
    # right_embedding = tf.slice(trans_embedding, [1, 0, 0], [1, -1, -1])
    # left_embedding = tf.reshape(left_embedding, [-1, word_dim])
    # right_embedding = tf.reshape(right_embedding, [-1, word_dim])
    # dot = tf.reduce_sum(tf.multiply(left_embedding, right_embedding), axis=1)
    # left_embedding_norm = tf.sqrt(tf.reduce_sum(tf.multiply(left_embedding, left_embedding), axis=1))
    # right_embedding_norm = tf.sqrt(tf.reduce_sum(tf.multiply(right_embedding, right_embedding), axis=1))
    # pred = dot / (left_embedding_norm * right_embedding_norm)
    # pred = tf.reshape(pred, [-1, 1])
    # sub = pred - y
    # loss = tf.reduce_mean(sub * sub)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)



    sess.run(tf.global_variables_initializer())
    summary = tf.summary.FileWriter('/Users/nali/PycharmProjects/MachineLearningLaboratory/ATEC/data/tensorboard')
    summary.add_graph(sess.graph, global_step=1)
    obop = tf.summary.merge_all()
    for epoch in range(50):
        sta = [0] * 5
        for i in range(batch_num-20):
        #     results = sess.run([optimizer,pred],
        #                        feed_dict={x: train_X[i], y: train_Y[i]})
        #
        #     print results[-1]
            results = sess.run([net['loss'], net['opt'], net['pred']],
                               feed_dict={net['x']: train_X[i], net['y']: train_Y[i]})
            sta[0] += results[0]
            mea = statistic_correct(results[2], train_Y[i])
            for cnt in range(1, 5):
                sta[cnt] += mea[cnt - 1]
                # print '-',
        ops = sess.run(obop)
        summary.add_summary(ops, epoch)
        acc,precision,recall, f1 =measure(sta[1],sta[2],sta[3],sta[4])
        print 'epoch-> ',epoch,'loss:', 1.0 * sta[0] / batch_num
        print '\tacc:', acc,
        print '\tprecision:', precision,
        print '\trecall:', recall,
        print '\tf1:', f1

    print 'TEST'
    results = sess.run([net['pred']],
                               feed_dict={net['x']: test_X})
    a,b,c,d = statistic_correct(results[0], test_Y)
    acc, precision, recall, f1 = measure(a,b,c,d)
    print '\tacc:', acc,
    print '\tprecision:', precision,
    print '\trecall:', recall,
    print '\tf1:', f1

    saver = tf.train.Saver()
    saver.save(sess, '/Users/nali/PycharmProjects/MachineLearningLaboratory/ATEC/data/mode_save/sum_word_embedding')
