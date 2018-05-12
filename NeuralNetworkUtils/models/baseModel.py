# encoding:utf-8



class BaseNeuralNetword:
    def predict(self, X):
        pass

    def train(self, X, Y, batch_size=10000, epoch=100):
        pass

    def createModel(self, *args, **kwargs):
        pass

    def saveModel(self, sess, file):
        pass

    def loadModel(self, sess, file):
        pass
