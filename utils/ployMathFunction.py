# coding:utf-8


import matplotlib.pyplot as plt
import numpy as np
import mathFunction as mf


def painFunction(f, x):
    plt.plot(x, f(x))
    plt.show()


x=np.linspace(0,100000,1000)
y=np.sqrt(x)

plt.plot(x, y)
plt.show()
#plt.scatter(x,mf.sigmoid(x))
#plt.show()