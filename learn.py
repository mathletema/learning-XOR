import numpy as np
import random
import pickle

runs = 5
passes = 300
search_space = 200
lr = 0.03

train_data = np.loadtxt('learn.csv', delimiter=',', skiprows=1)
train_num = test_num = train_data.shape[0]

test_data = np.loadtxt('verification.csv', delimiter=',', skiprows=1)

def random_vec():
    ran_vec = np.empty(9)
    for i in range(9):
        ran_vec[i] = random.random() - 0.5
    length = np.linalg.norm(ran_vec)
    ran_vec = ran_vec / length
    return (np.array([ran_vec[0:2], ran_vec[2:4]]),
            np.array([ran_vec[4:5], ran_vec[5:6]]),
            np.array([ran_vec[6:7], ran_vec[7:8]]),
            ran_vec[8])

def f(x, W, c, w, b):
    a = np.maximum(np.matmul(W.T, x) + c, 0)
    return np.matmul(w.T, a) + b

def loss_test(W, c, w, b):
    x = test_data[:, :2].T
    y = test_data[:, 2:3].T
    return np.sum(np.square(y - f(x, W, c, w, b))) / 500

def loss_train(W, c, w, b):
    x = train_data[:, :2].T
    y = train_data[:, 2:3].T
    return np.sum(np.square(y - f(x, W, c, w, b))) / 500

def main():
    
    params = []

    for u in range(runs):
        k = random_vec()
        W = k[0]
        c = np.array([[0], [0]])
        w = k[2]
        b = 0
        for i in range(passes):
            print("Train loss:", loss_train(W, c, w, b))
            for j in range(search_space):
                W1, c1, w1, b1 = [x[0] + lr * x[1] for x in zip((W, c, w, b), random_vec())]
                if (loss_train(W1, c1, w1, b1)) < loss_train(W, c, w, b):
                    W, c, w, b = W1, c1, w1, b1
        test_loss = loss_test(W, c, w, b)
        print("Try", u+1, ", test loss:", test_loss)
        print("\n\n\n\n")
        params.append((W, c, w, b, test_loss))

    for i in range(runs):
        print("Try", i+1, ", test loss:", params[i][4])

    W, c, w, b, _ = min(params, key=lambda tup: tup[4])

    with open('params.bin', 'wb') as f:
        pickle.dump((W, c, w, b), f)


if __name__ == "__main__":
    main()
