#coding:utf-8

import tensorflow as tf
import matplotlib.pyplot as plt


def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)


def dense_layer(value, weightShape, outputShape, name=None, active= 'relu'):

    weight = weight_variable(shape=weightShape)
    bias = bias_variable(shape=outputShape)
    if active=='relu':
        hidden = tf.nn.relu(tf.matmul(value, weight) + bias, name=name)
    elif active=='linear':
        hidden = tf.matmul(value, weight) + bias
    else:
        hidden = tf.nn.sigmoid(tf.matmul(value, weight) + bias, name=name)

    return hidden

x = tf.placeholder(dtype=tf.float32,shape=[None, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

hidden1 = dense_layer(x,[1,10],[10], active='sigmoid')
# hidden2 = dense_layer(hidden1,[10,20],[20])
# hidden3 = dense_layer(hidden2,[20,10],[10])
# hidden4 = dense_layer(x,[10,1],[1])
pred = dense_layer(hidden1,[10,1],[1],'linear')

sub = pred - y
loss = tf.nn.l2_loss(sub)
adam = tf.train.AdamOptimizer(learning_rate=0.1)
opti =adam.minimize(loss)


def trainPredict(sess,X, Y, first=False):
    X = [[i] for i in X]
    Y = [[i] for i in Y]
    if sess is None:
        sess = tf.Session()
    if first:
        sess.run(tf.global_variables_initializer())
    for i in range(500):
        for k in range(len(X)):
            res = sess.run([opti, loss],feed_dict={x:X,y:Y})
        # print res[1]


    predX = [[float(i)] for i in range(int(X[-1][0])+5)]
    ypred = sess.run(pred, feed_dict={x:predX})
    return [i[0] for i in predX], [i[0] for i in ypred]

X=[]
Y=[]
#
X = [float(i) for i in range(200)]
Y = [ k*(k-10) for k in X]
xp,yp = trainPredict(None, X,Y, True)
print xp
print yp
plt.scatter(X,Y)
plt.plot(xp,yp)
plt.show()
#
# first = True
# sess = tf.Session()
# while 1:
#     str =raw_input('输入x,y:')
#     xy = str.strip().split(',')
#     x0=xy[0]
#     y0=xy[1]
#     X.append(float(x0))
#     Y.append(float(y0))
#     xp,yp = trainPredict(sess, X,Y, first)
#     first = False
#     print xp
#     print yp
#     plt.scatter(X,Y)
#     plt.plot(xp,yp)
#     plt.show()