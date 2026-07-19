import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(data, label, lr=0.01, steps=1000):
    N, D = data.shape
    W = np.zeros(D)
    b = 0.0
    loss_list = []
    
    for i in range(steps):
        z = np.dot(data, W) + b
        y = sigmoid(z)
        loss = -np.mean(label * np.log(y + 1e-15) + (1 - label) * np.log(1 - y + 1e-15))
        loss_list.append(loss)
        error = y - label
        dw = (1 / N) * np.dot(data.T, error)
        db = (1 / N) * np.sum(error)
        W = W - lr * dw
        b = b - lr * db
        
    return W, b