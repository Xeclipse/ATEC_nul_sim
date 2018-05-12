import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMCell
from tensorFlowTest.libs.utils import weight_variable, bias_variable
from tensorflow.python.ops import math_ops
from tensorflow.contrib import layers as la


def embedding_layer(value, vocabDim, vecDim):
    with tf.name_scope("embedding"):
        embeddingVector = la.embed_sequence(value, vocabDim, vecDim,
                                            initializer=tf.initializers.random_normal(mean=0, stddev=0.01),
                                            regularizer=la.l2_regularizer(scale=0.1))
        return embeddingVector


def dense_layer(value, weightShape, outputShape, name=None, active= 'relu'):
    with tf.name_scope("dense"):
        weight = weight_variable(shape=weightShape)
        bias = bias_variable(shape=outputShape)
        if active=='relu':
            hidden = tf.nn.relu(tf.matmul(value, weight) + bias, name=name)
        else:
            hidden = tf.nn.sigmoid(tf.matmul(value, weight) + bias, name=name)
        tf.summary.histogram(name='denseW', values=weight)
        tf.summary.histogram(name='denseBias', values=bias)
        return hidden


def conv_1d_layer(value, filterSize, nFilter, stride, channels):
    with tf.name_scope("conv1d"):
        stride = stride
        filter_size = filterSize
        n_filter = nFilter
        wConvLayer = weight_variable([filter_size, channels, n_filter])
        bConvLayer = bias_variable([n_filter])
        hidden = tf.nn.relu(
            tf.nn.conv1d(
                value=value,
                filters=wConvLayer,
                stride=stride,
                padding='SAME') +
            bConvLayer)
        tf.summary.histogram(name='convW', values=wConvLayer)
        tf.summary.histogram(name='convBias', values=bConvLayer)
        return hidden


def softmax_layer(value, inputShape, outputShape, name = None):
    with tf.name_scope("softmax"):
        wSoftmax = weight_variable(shape=inputShape)
        bSoftmax = bias_variable(shape=outputShape)
        ypred = tf.nn.softmax(tf.matmul(value, wSoftmax) + bSoftmax, name=name)
        tf.summary.histogram(name='softmaxW', values=wSoftmax)
        tf.summary.histogram(name='softmaxB', values=bSoftmax)
        return ypred


def dropout_layer(value):
    with tf.name_scope("dropout"):
        dropout = tf.nn.dropout(x=value, keep_prob=1.0)
        return dropout

def attention_layer(value, hiddenUnits, name):
    with tf.name_scope("attention_layer"):
        with tf.variable_scope("rnn_attention_" + name, reuse=False):
            with tf.variable_scope("rnn_encoder_variable", reuse=False):
                cell = LSTMCell(num_units=hiddenUnits, initializer=tf.truncated_normal_initializer)
                outputs, lastStates = dynamicRnnLayer(value, cell)
            with tf.variable_scope("rnn_attention_variable", reuse=False):
                attentionCell = LSTMCell(num_units=1, initializer=tf.truncated_normal_initializer)
                attentions, lastAttentionState = dynamicRnnLayer(outputs, attentionCell)
        attentionEmbeddings = tf.multiply(outputs, attentions)
        return attentionEmbeddings


def l1_normalize(value, dim=1, epsilon=1e-12, name=None):
    with tf.name_scope("l1_normalize"):
        valuesum = tf.reduce_sum(value,dim,keep_dims=True)
        inv_norm = tf.reciprocal(math_ops.maximum(valuesum, epsilon))
        normalize = tf.multiply(value, inv_norm, name=name)
        return normalize

def optimize_op(value, name = 'adagrad',rate = 0.01):
    with tf.name_scope("optimizer"):
        optimizer = None
        if name =='adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=rate)
        if name =='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        if name == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=rate)
        if name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=rate)
        if name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        return optimizer.minimize(value, name=name+'_optimizer')

def optimize_op_rsrp(value, rate = 0.01):
    with tf.name_scope("optimizer"):
        rsrp = tf.train.RMSPropOptimizer(learning_rate=rate)
        return rsrp.minimize(value, name='rsrp_optimizer')

def flatten(value, flattenShape):
    with tf.name_scope("flatten"):
        return tf.reshape(value, shape=flattenShape)


def accuracy(ypred, y, name=None):
    with tf.name_scope("measure"):
        correct_prediction = tf.equal(tf.argmax(ypred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy_'+name)
        tf.summary.scalar(name="accuracy", tensor=acc)
        return acc


def cost_loss(ypred, y):
    with tf.name_scope("loss"):
        sub = ypred - y
        pow2 = tf.multiply(sub, sub, name="pow2")
        L2 = tf.reduce_mean(input_tensor=pow2, axis=1, keep_dims=True)
        costMatrix = tf.constant([[5], [1]], dtype=tf.float32)
        costList = tf.matmul(y, costMatrix)
        loss = tf.reduce_mean(tf.matmul(L2, costList, transpose_a=True), name='cost_loss')
        tf.summary.scalar(name="cost_loss", tensor=loss)
        return loss

def cost_loss_multi(ypred, y, weightOne =2.0):
    with tf.name_scope("loss"):
        sub = ypred - y
        weight =  tf.add(tf.scalar_mul(weightOne, y),1)
        weightPow2 = tf.multiply(tf.multiply(sub, sub), weight)
        L2 = tf.reduce_sum(input_tensor=weightPow2, axis=1)
        loss = tf.reduce_mean(L2, name='cost_loss')
        tf.summary.scalar(name="cost_loss", tensor=loss)
        return loss

def cross_entropy_loss(ypred, y):
    # tf.nn.sigmoid_cross_entropy_with_logits()
    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y,tf.log(ypred)), name="cross_entropy")
        tf.summary.scalar(name="cross_entropy", tensor=cost)
        return cost


def conv_flatten_dense_layer(value, filtersize, nFilter, stride, channels, outdim):
    sendim = int(value.get_shape().as_list()[1])
    with tf.name_scope("packageLayer"):
        hidden = conv_1d_layer(value=value, filterSize=filtersize, nFilter=nFilter, stride=stride, channels=channels)
        dim = int(tf.math.ceil(1.0 * sendim / stride)) * nFilter
        flat = flatten(value=hidden, flattenShape=[-1, dim])
        dense = dense_layer(value=flat, weightShape=[dim, outdim], outputShape=[outdim])
        return dense

def square_loss(ypred, y):
    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.reduce_sum(tf.multiply( (y-ypred), (y-ypred)), axis=1), name="square_loss")
        tf.summary.scalar(name="square_loss", tensor=cost)
        return cost

def dynamicRnnLayer(value, cell, lengths = None, name=None):
    with tf.name_scope("RNN"):
        if lengths is not None:
            return tf.nn.dynamic_rnn(cell=cell, inputs=value, sequence_length=lengths, dtype=tf.float32)
        else:
            return tf.nn.dynamic_rnn(cell=cell, inputs=value, dtype=tf.float32)

def MultiCNN(value, vecDim, nFilter = 30):
    # process titles
    convEmbedding2 = conv_1d_layer(value, filterSize=2, nFilter=nFilter, stride=1, channels=vecDim)
    convEmbedding4 = conv_1d_layer(value, filterSize=4, nFilter=nFilter, stride=1, channels=vecDim)
    convEmbedding8 = conv_1d_layer(value, filterSize=8, nFilter=nFilter, stride=1, channels=vecDim)
    # convEmbedding8 = conv_1d_layer(value, filterSize=8, nFilter=nFilter, stride=1, channels=vecDim)
    # convEmbedding16 = conv_1d_layer(value, filterSize=16, nFilter=nFilter, stride=1, channels=vecDim)
    embedddingSum = tf.nn.l2_normalize(tf.reduce_sum(value,axis=1), dim = 1)
    convEmbedding2 = tf.nn.l2_normalize(tf.reduce_sum(convEmbedding2, axis=1), dim=1)
    convEmbedding4 = tf.nn.l2_normalize(tf.reduce_sum(convEmbedding4, axis=1),dim=1)
    convEmbedding8 = tf.nn.l2_normalize(tf.reduce_sum(convEmbedding8, axis=1), dim=1)
    return tf.concat([embedddingSum, convEmbedding2,convEmbedding4,convEmbedding8], axis=1)