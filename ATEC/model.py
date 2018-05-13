# coding:utf-8
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

def embedding_sum_model(sen_dim, vocab_dim, word_dim):
    x = tf.placeholder(dtype=tf.int32, shape=[None, 2, sen_dim], name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="label")

    intialembedding = tf.random_normal(shape=[vocab_dim, word_dim], mean=0.0, stddev=0.1)
    word_vector_table = tf.Variable(intialembedding, name='word_vector')
    regulizer = l2_regularizer(0.5)
    loss_regularizer = regulizer(word_vector_table)
    tf.summary.histogram(name='embedding', values=word_vector_table)

    # all_embedding -> [batch, 2, sen_dim, word_dim]
    all_embedding = tf.nn.embedding_lookup(word_vector_table, x)
    # sum_embedding ->[batch, 2, word_dim]
    sum_embedding = tf.reduce_sum(all_embedding, axis=2)
    # normalize_embedding->[batch, 2, word_dim]
    normalize_embedding = tf.nn.l2_normalize(sum_embedding, dim=1)
    # calc cosine distance
    trans_embedding = tf.transpose(normalize_embedding, perm=[1, 0, 2])
    left_embedding = tf.slice(trans_embedding, [0, 0, 0], [1, -1, -1])
    right_embedding = tf.slice(trans_embedding, [1, 0, 0], [1, -1, -1])
    left_embedding = tf.reshape(left_embedding, [-1, word_dim])
    right_embedding = tf.reshape(right_embedding, [-1, word_dim])
    dot = tf.reduce_sum(tf.multiply(left_embedding, right_embedding), axis=1)
    left_embedding_norm = tf.sqrt(tf.reduce_sum(tf.multiply(left_embedding, left_embedding), axis=1))
    right_embedding_norm = tf.sqrt(tf.reduce_sum(tf.multiply(right_embedding, right_embedding), axis=1))
    distance = dot / (left_embedding_norm * right_embedding_norm)
    distance = tf.reshape(distance, [-1, 1])
    pred = tf.layers.dense(distance, units=2, activation=tf.nn.sigmoid)
    loss = tf.losses.mean_squared_error(y, pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return {
        'x': x,
        'y': y,
        'loss': loss,
        'pred': pred,
        'opt': optimizer
    }


def embedding_sum_model_square_distance(sen_dim, vocab_dim, word_dim):
    x = tf.placeholder(dtype=tf.int32, shape=[None, 2, sen_dim], name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="label")

    intialembedding = tf.random_normal(shape=[vocab_dim, word_dim], mean=0.0, stddev=0.1)
    word_vector_table = tf.Variable(intialembedding, name='word_vector')
    # regulizer = l2_regularizer(0.5)
    # loss_regularizer = regulizer(word_vector_table)
    # indices = [[0,i] for i in range(word_dim)]
    # indices.extend([[1,i] for i in range(word_dim)])
    # assign_value = [0.0] * (word_dim*2)
    # word_vector_table = tf.scatter_nd_update(word_vector_table,indices,assign_value)
    tf.summary.histogram(name='word_vector', values=word_vector_table)

    no_info_vector = tf.slice(word_vector_table,[0,0],[2,-1])
    loss_no_info = tf.reduce_sum(no_info_vector*no_info_vector)
    # norm_word_embedding = tf.slice(word_vector_table,[2,0],[vocab_dim-2, word_dim])
    # norm_sub = tf.reduce_sum(norm_word_embedding*norm_word_embedding, 1)-1.0
    # loss_norm = tf.reduce_mean(norm_sub*norm_sub)


    # all_embedding -> [batch, 2, sen_dim, word_dim]
    all_embedding = tf.nn.embedding_lookup(word_vector_table, x)
    # sum_embedding ->[batch, 2, word_dim]
    sum_embedding = tf.reduce_sum(all_embedding, axis=2)
    # normalize_embedding->[batch, 2, word_dim]
    normalize_embedding = tf.nn.l2_normalize(sum_embedding, dim=1)
    # calc square distance
    trans_embedding = tf.transpose(normalize_embedding, perm=[1, 0, 2])
    left_embedding = tf.slice(trans_embedding, [0, 0, 0], [1, -1, -1])
    right_embedding = tf.slice(trans_embedding, [1, 0, 0], [1, -1, -1])
    left_embedding = tf.reshape(left_embedding, [-1, word_dim])
    right_embedding = tf.reshape(right_embedding, [-1, word_dim])
    sub = left_embedding-right_embedding
    distance = tf.reduce_sum(sub*sub, 1)
    distance = tf.reshape(distance, [-1, 1])
    pred = tf.layers.dense(distance,units=2, activation=tf.nn.sigmoid)
    loss = tf.losses.mean_squared_error(y,pred)+loss_no_info#+loss_norm
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return {
        'x': x,
        'y': y,
        'loss': loss,
        'pred': pred,
        'opt': optimizer
    }



def embedding_cnn_model(sen_dim, vocab_dim, word_dim):
    x = tf.placeholder(dtype=tf.int32, shape=[None, 2, sen_dim], name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="label")

    intialembedding = tf.random_normal(shape=[vocab_dim, word_dim], mean=0.0, stddev=0.1)
    word_vector_table = tf.Variable(intialembedding, name='word_vector')
    regulizer = l2_regularizer(0.5)
    loss_regularizer = regulizer(word_vector_table)
    tf.summary.histogram(name='embedding', values=word_vector_table)

    # all_embedding -> [batch, 2, sen_dim, word_dim]
    all_embedding = tf.nn.embedding_lookup(word_vector_table, x)
    # sum_embedding ->[batch, 2, word_dim]




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
    # distance = dot / (left_embedding_norm * right_embedding_norm)
    # distance = tf.reshape(distance, [-1, 1])
    # pred = tf.layers.dense(distance, units=2, activation=tf.nn.sigmoid)
    # loss = tf.losses.mean_squared_error(y, pred)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    #
    # return {
    #     'x': x,
    #     'y': y,
    #     'loss': loss,
    #     'pred': pred,
    #     'opt': optimizer
    # }