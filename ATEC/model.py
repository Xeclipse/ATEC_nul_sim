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

    no_info_vector = tf.slice(word_vector_table, [0, 0], [2, -1])
    loss_no_info = tf.reduce_sum(no_info_vector * no_info_vector)
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
    sub = left_embedding - right_embedding
    distance = tf.reduce_sum(sub * sub, 1)
    distance = tf.reshape(distance, [-1, 1])
    pred = tf.layers.dense(distance, units=2, activation=tf.nn.sigmoid)
    loss = tf.losses.mean_squared_error(y, pred) + loss_no_info  # +loss_norm
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return {
        'x': x,
        'y': y,
        'loss': loss,
        'pred': pred,
        'opt': optimizer
    }


def embedding_sum_model_square_distance_v2(sen_dim, vocab_dim, word_dim):
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

    no_info_vector = tf.slice(word_vector_table, [0, 0], [1, -1])
    loss_no_info = tf.reduce_sum(no_info_vector * no_info_vector)
    norm_word_embedding = tf.slice(word_vector_table, [1, 0], [vocab_dim - 1, word_dim])
    norm_sub = tf.reduce_sum(norm_word_embedding * norm_word_embedding, 1) - 1.0
    loss_norm = tf.reduce_mean(norm_sub * norm_sub)

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
    sub = left_embedding - right_embedding
    distance = tf.reduce_sum(sub * sub, 1)
    distance = tf.reshape(distance, [-1, 1])
    pred = tf.layers.dense(distance, units=2, activation=tf.nn.sigmoid)
    loss = tf.losses.mean_squared_error(y, pred) + loss_no_info  # +0.1*loss_norm
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return {
        'x': x,
        'y': y,
        'loss': loss,
        'pred': pred,
        'opt': optimizer
    }


#
# def left_conv_embedding(left_embeddings):


def embedding_cnn_model_square_distance(sen_dim, vocab_dim, word_dim):
    x = tf.placeholder(dtype=tf.int32, shape=[None, 2, sen_dim], name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="label")

    intialembedding = tf.random_normal(shape=[vocab_dim, word_dim], mean=0.0, stddev=0.1)
    word_vector_table = tf.Variable(intialembedding, name='word_vector')
    tf.summary.histogram(name='embedding', values=word_vector_table)
    no_info_vector = tf.slice(word_vector_table, [0, 0], [1, -1])
    loss_no_info = tf.reduce_sum(no_info_vector * no_info_vector)
    norm_word_embedding = tf.slice(word_vector_table,[1,0],[vocab_dim-1, word_dim])
    norm_sub = tf.reduce_sum(norm_word_embedding*norm_word_embedding, 1)-1.0
    loss_norm = tf.reduce_mean(norm_sub*norm_sub)



    all_embedding = tf.nn.embedding_lookup(word_vector_table, x)
    left_embeddings = tf.slice(all_embedding, [0, 0, 0, 0], [-1, 1, -1, -1])
    right_embeddings = tf.slice(all_embedding, [0, 1, 0, 0], [-1, 1, -1, -1])

    left_embeddings = tf.reshape(left_embeddings, [-1, sen_dim, word_dim])
    left_conv_embeddings_2 = tf.layers.conv1d(left_embeddings, filters=40, kernel_size=3, strides=1,
                                              activation=tf.nn.relu,
                                              kernel_regularizer=l2_regularizer(0.5),
                                              bias_regularizer=l2_regularizer(0.5),
                                              kernel_initializer=tf.glorot_normal_initializer(),
                                              bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                                              name='conv_1d', reuse=None)
    left_conv_embeddings_2_residual = tf.layers.conv1d(left_conv_embeddings_2, filters=40, kernel_size=3, strides=1,
                                              activation=tf.nn.relu,
                                              kernel_regularizer=l2_regularizer(0.5),
                                              bias_regularizer=l2_regularizer(0.5),
                                              kernel_initializer=tf.glorot_normal_initializer(),
                                              bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                                              name='conv_1d_2', reuse=None)
    left_conv_embeddings_2 = left_conv_embeddings_2_residual+left_conv_embeddings_2_residual
    left_embedding = tf.reduce_sum(left_conv_embeddings_2, axis=1)
    left_embedding = tf.layers.dense(left_embedding, 40, activation=tf.nn.relu, name='projection',
                                             reuse=None)



    right_embeddings = tf.reshape(right_embeddings, [-1, sen_dim, word_dim])
    right_conv_embeddings_2 = tf.layers.conv1d(right_embeddings, filters=40, kernel_size=3, strides=1,
                                               activation=tf.nn.relu,
                                               kernel_regularizer=l2_regularizer(0.5),
                                               bias_regularizer=l2_regularizer(0.5),
                                               kernel_initializer=tf.glorot_normal_initializer(),
                                               bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                                               name='conv_1d', reuse=True)
    right_conv_embeddings_residual = tf.layers.conv1d(right_conv_embeddings_2, filters=40, kernel_size=3, strides=1,
                                               activation=tf.nn.relu,
                                               kernel_regularizer=l2_regularizer(0.5),
                                               bias_regularizer=l2_regularizer(0.5),
                                               kernel_initializer=tf.glorot_normal_initializer(),
                                               bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                                               name='conv_1d_2', reuse=True)
    right_conv_embeddings_2 = right_conv_embeddings_2+right_conv_embeddings_residual
    right_embedding = tf.reduce_sum(right_conv_embeddings_2, axis=1)
    right_embedding = tf.layers.dense(right_embedding, 40, activation=tf.nn.relu, name='projection',
                                     reuse=True)



    regulizer_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    sub = left_embedding - right_embedding
    distance = tf.reduce_sum(sub * sub, 1)
    distance = tf.reshape(distance, [-1, 1])
    pred = tf.layers.dense(distance, units=2, activation=tf.nn.sigmoid)
    loss = tf.losses.mean_squared_error(y, pred) + loss_no_info +0.1*loss_norm+ 0.007*regulizer_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return {
        'x': x,
        'y': y,
        'loss': loss,
        'pred': pred,
        'opt': optimizer
    }


def embedding_cnn_model_projection_distance(sen_dim, vocab_dim, word_dim):
    x = tf.placeholder(dtype=tf.int32, shape=[None, 2, sen_dim], name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="label")

    intialembedding = tf.random_normal(shape=[vocab_dim, word_dim], mean=0.0, stddev=0.1)
    word_vector_table = tf.Variable(intialembedding, name='word_vector')
    tf.summary.histogram(name='embedding', values=word_vector_table)
    no_info_vector = tf.slice(word_vector_table, [0, 0], [1, -1])
    loss_no_info = tf.reduce_sum(no_info_vector * no_info_vector)
    norm_word_embedding = tf.slice(word_vector_table,[1,0],[vocab_dim-1, word_dim])
    norm_sub = tf.reduce_sum(norm_word_embedding*norm_word_embedding, 1)-1.0
    loss_norm = tf.reduce_mean(norm_sub*norm_sub)



    all_embedding = tf.nn.embedding_lookup(word_vector_table, x)
    left_embeddings = tf.slice(all_embedding, [0, 0, 0, 0], [-1, 1, -1, -1])
    right_embeddings = tf.slice(all_embedding, [0, 1, 0, 0], [-1, 1, -1, -1])

    left_embeddings = tf.reshape(left_embeddings, [-1, sen_dim, word_dim])
    left_conv_embeddings_2 = tf.layers.conv1d(left_embeddings, filters=40, kernel_size=3, strides=1,
                                              activation=tf.nn.relu,
                                              kernel_regularizer=l2_regularizer(0.5),
                                              bias_regularizer=l2_regularizer(0.5),
                                              kernel_initializer=tf.glorot_normal_initializer(),
                                              bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                                              name='conv_1d', reuse=None)
    left_conv_embeddings_2_residual = tf.layers.conv1d(left_conv_embeddings_2, filters=40, kernel_size=3, strides=1,
                                              activation=tf.nn.relu,
                                              kernel_regularizer=l2_regularizer(0.5),
                                              bias_regularizer=l2_regularizer(0.5),
                                              kernel_initializer=tf.glorot_normal_initializer(),
                                              bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                                              name='conv_1d_2', reuse=None)
    left_conv_embeddings_2 = left_conv_embeddings_2_residual+left_conv_embeddings_2_residual
    left_embedding = tf.reduce_sum(left_conv_embeddings_2, axis=1)



    right_embeddings = tf.reshape(right_embeddings, [-1, sen_dim, word_dim])
    right_conv_embeddings_2 = tf.layers.conv1d(right_embeddings, filters=40, kernel_size=3, strides=1,
                                               activation=tf.nn.relu,
                                               kernel_regularizer=l2_regularizer(0.5),
                                               bias_regularizer=l2_regularizer(0.5),
                                               kernel_initializer=tf.glorot_normal_initializer(),
                                               bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                                               name='conv_1d', reuse=True)
    right_conv_embeddings_residual = tf.layers.conv1d(right_conv_embeddings_2, filters=40, kernel_size=3, strides=1,
                                               activation=tf.nn.relu,
                                               kernel_regularizer=l2_regularizer(0.5),
                                               bias_regularizer=l2_regularizer(0.5),
                                               kernel_initializer=tf.glorot_normal_initializer(),
                                               bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                                               name='conv_1d_2', reuse=True)
    right_conv_embeddings_2 = right_conv_embeddings_2+right_conv_embeddings_residual
    right_embedding = tf.reduce_sum(right_conv_embeddings_2, axis=1)


    concat_embedding = tf.concat([left_embedding, right_embedding], axis=1)

    dense_1 = tf.layers.dense(concat_embedding,units=50, activation=tf.nn.relu)
    dense_2 = tf.layers.dense(dense_1, units=10, activation=tf.nn.relu)

    regulizer_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    pred = tf.layers.dense(dense_2, units=2, activation=tf.nn.sigmoid)
    loss = tf.losses.mean_squared_error(y, pred) + loss_no_info +0.1*loss_norm+ 0.007*regulizer_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return {
        'x': x,
        'y': y,
        'loss': loss,
        'pred': pred,
        'opt': optimizer
    }




def embedding_summing_cnn_pyramid(input, reuse=None):
    conv_1 = tf.layers.conv1d(input, filters=40, kernel_size=3, strides=1,
                     activation=tf.nn.relu,
                     kernel_regularizer=l2_regularizer(0.5),
                     bias_regularizer=l2_regularizer(0.5),
                     kernel_initializer=tf.glorot_normal_initializer(),
                     bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                     name='conv_1d', reuse=reuse)
    conv_2 = tf.layers.conv1d(conv_1, filters=20, kernel_size=3, strides=1,
                     activation=tf.nn.relu,
                     kernel_regularizer=l2_regularizer(0.5),
                     bias_regularizer=l2_regularizer(0.5),
                     kernel_initializer=tf.glorot_normal_initializer(),
                     bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                     name='conv_1d2', reuse=reuse)
    conv_3 = tf.layers.conv1d(conv_2, filters=5, kernel_size=3, strides=1,
                     activation=tf.nn.relu,
                     kernel_regularizer=l2_regularizer(0.5),
                     bias_regularizer=l2_regularizer(0.5),
                     kernel_initializer=tf.glorot_normal_initializer(),
                     bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                     name='conv_1d3', reuse=reuse)
    ret = tf.reduce_sum(conv_3,axis=1)
    return ret


def embedding_summing_cnn_pyramid(input, reuse=None):
    conv_1 = tf.layers.conv1d(input, filters=40, kernel_size=3, strides=1,
                     activation=tf.nn.relu,
                     kernel_regularizer=l2_regularizer(0.5),
                     bias_regularizer=l2_regularizer(0.5),
                     kernel_initializer=tf.glorot_normal_initializer(),
                     bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                     name='conv_1d', reuse=reuse)
    conv_2 = tf.layers.conv1d(conv_1, filters=20, kernel_size=3, strides=1,
                     activation=tf.nn.relu,
                     kernel_regularizer=l2_regularizer(0.5),
                     bias_regularizer=l2_regularizer(0.5),
                     kernel_initializer=tf.glorot_normal_initializer(),
                     bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                     name='conv_1d2', reuse=reuse)
    conv_3 = tf.layers.conv1d(conv_2, filters=5, kernel_size=3, strides=1,
                     activation=tf.nn.relu,
                     kernel_regularizer=l2_regularizer(0.5),
                     bias_regularizer=l2_regularizer(0.5),
                     kernel_initializer=tf.glorot_normal_initializer(),
                     bias_initializer=tf.glorot_normal_initializer(), padding='SAME',
                     name='conv_1d3', reuse=reuse)
    ret = tf.reduce_sum(conv_3,axis=1)
    return ret


def embedding_hierarchical_cnn_model_distance(sen_dim, vocab_dim, word_dim):
    x = tf.placeholder(dtype=tf.int32, shape=[None, 2, sen_dim], name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="label")

    intialembedding = tf.random_normal(shape=[vocab_dim, word_dim], mean=0.0, stddev=0.1)
    word_vector_table = tf.Variable(intialembedding, name='word_vector')
    tf.summary.histogram(name='embedding', values=word_vector_table)
    no_info_vector = tf.slice(word_vector_table, [0, 0], [1, -1])
    loss_no_info = tf.reduce_sum(no_info_vector * no_info_vector)
    norm_word_embedding = tf.slice(word_vector_table,[1,0],[vocab_dim-1, word_dim])
    norm_sub = tf.reduce_sum(norm_word_embedding*norm_word_embedding, 1)-1.0
    loss_norm = tf.reduce_mean(norm_sub*norm_sub)


    all_embedding = tf.nn.embedding_lookup(word_vector_table, x)
    left_embeddings = tf.slice(all_embedding, [0, 0, 0, 0], [-1, 1, -1, -1])
    right_embeddings = tf.slice(all_embedding, [0, 1, 0, 0], [-1, 1, -1, -1])

    left_embeddings = tf.reshape(left_embeddings,[-1,sen_dim,word_dim])
    left_embedding = embedding_summing_cnn_pyramid(left_embeddings)

    right_embeddings = tf.reshape(right_embeddings, [-1, sen_dim, word_dim])
    right_embedding = embedding_summing_cnn_pyramid(right_embeddings, reuse=True)

    sub = left_embedding-right_embedding
    sub_2 = sub*sub

    regulizer_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    dense = tf.layers.dense(sub_2, units=2, activation=tf.nn.sigmoid)
    pred = tf.nn.softmax(dense, axis=1)
    loss = tf.losses.mean_squared_error(y, pred) + loss_no_info #+0.1*loss_norm+ 0.007*regulizer_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return {
        'x': x,
        'y': y,
        'loss': loss,
        'pred': pred,
        'opt': optimizer
    }



def cnn_layer(inputs):
    conv = tf.layers.conv1d(inputs, filters=40, kernel_size=5, strides=1, activation=tf.nn.relu,
                            kernel_regularizer=l2_regularizer(1.0), bias_regularizer=l2_regularizer(1.0))
