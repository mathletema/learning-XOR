import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle


with open('params.bin', 'rb') as f:
    W, c, w, b = pickle.load(f)

def f_given(x, W, c, w, b):
    a = np.maximum(np.matmul(W.T, x) + c, 0)
    return np.matmul(w.T, a) + b

def f(x, y):
    return f_given([[x],[y]], W, c, w, b)[0, 0]

data = np.loadtxt('verification.csv', delimiter=',', skiprows=1)

x1 = data[:,0]
x2 = data[:,1]
color = data[:, 2]

x = np.arange(-1.0, 2.0, 0.1)
y = np.arange(-1.0, 2.0, 0.1)

X, Y = np.meshgrid(x, y)
Z = np.empty((30, 30))

rgb = plt.get_cmap('jet')(color)

def plot(f):
    for i in range(30):
        for j in range(30):
            Z[i, j] = 1 -  f(x[i], y[j])

    plt.pcolormesh(X, Y, Z, cmap='RdBu', shading='gouraud', norm=colors.TwoSlopeNorm(0.5))
    plt.scatter(x1, x2, color = rgb, s=2)
    plt.savefig('out.png')
    plt.show()
plot(f)