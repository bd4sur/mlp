# forked from https://github.com/dennybritz/nn-from-scratch
# ref https://dennybritz.com/posts/wildml/implementing-a-neural-network-from-scratch/
# BD4SUR 2023.10

import math
import random
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
frames = []

class Config:
    batch_size = 64
    learning_rate = 0.01
    reg_lambda = 0.01

def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(1024, noise=0.2)
    return X, y

def capture_chart(X, y, model, epoch, loss):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict(model, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    contour = ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    title_str = "Loss @ epoch %i : %f" % (epoch, loss)
    title = plt.text(0, 0, title_str, horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
    frame = []
    frame.extend(contour.collections)
    frame.append(scatter)
    frame.append(title)
    frames.append(frame)

def forward(x, model):
    W, b, af = model["W"], model["b"], model["activation_function"]
    num_hidden_layers = len(W)
    a = [0] * (num_hidden_layers + 1)
    z = [0] * (num_hidden_layers + 1)
    a[0] = x
    for i in range(num_hidden_layers):
        Wi = W[i]
        bi = b[i]
        z[i+1] = a[i].dot(Wi) + bi
        # 激活函数
        if af == "tanh":
            a[i+1] = np.tanh(z[i+1])        # tanh
        elif af == "ReLU":
            a[i+1] = np.maximum(z[i+1], 0)  # ReLU
        else:
            raise
    # softmax
    exp_scores = np.exp(z[num_hidden_layers])
    outputs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return (outputs, a)

def dReLU(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def bp(outputs, y, a, model):
    W, af = model["W"], model["activation_function"]
    num_hidden_layers = len(W)
    num_y = len(outputs)
    delta = outputs
    delta[range(num_y), y] -= 1  # 交叉熵
    dW = [0] * num_hidden_layers
    db = [0] * num_hidden_layers
    for i in range(num_hidden_layers):
        i = num_hidden_layers - i - 1
        dW[i] = (a[i].T).dot(delta) + Config.reg_lambda * W[i] # L2-正则化
        if i == num_hidden_layers - 1:
            db[i] = np.sum(delta, axis=0, keepdims=True)
        else:
            db[i] = np.sum(delta, axis=0)
        # 激活函数
        if af == "tanh":
            delta = delta.dot(W[i].T) * (1 - np.power(a[i], 2))  # tanh
        elif af == "ReLU":
            delta = delta.dot(W[i].T) * dReLU(a[i])              # ReLU
        else:
            raise
    return (dW, db)

def calculate_loss(model, X, y):
    x_num = len(X)
    W = model['W']
    outputs = forward(X, model)[0]
    data_loss = np.sum(-np.log(outputs[range(x_num), y]))
    # L2-正则化
    sum = 0
    for i in range(len(W)):
        sum += np.sum(np.square(W[i]))
    data_loss += Config.reg_lambda / 2 * sum
    return 1.0 / x_num * data_loss

def predict(model, x):
    output = forward(x, model)[0]
    return np.argmax(output, axis=1)

# layer_dims = [input_layer_dim, hidden1, hidden2, ... , output_layer_dim]
def init_model(layer_dims, activation_function="tanh"):
    num_hidden_layers = len(layer_dims) - 1
    np.random.seed(0)
    W = [0] * num_hidden_layers
    b = [0] * num_hidden_layers
    for layer_index in range(num_hidden_layers):
        W[layer_index] = np.random.randn(layer_dims[layer_index], layer_dims[layer_index + 1]) / np.sqrt(layer_dims[layer_index])
        b[layer_index] = np.zeros((1, layer_dims[layer_index + 1]))
    return { "W": W, "b": b, "num_hidden_layers": num_hidden_layers, "activation_function": activation_function }

def train(X, y, model, epoch_num=100):
    W, b, num_hidden_layers = model["W"], model["b"], model["num_hidden_layers"]
    training_set_size = len(X)
    # 完成一个epoch所需的iteration数
    iteration_num = math.ceil(training_set_size / Config.batch_size)
    for epoch in range(epoch_num):
        # 打乱训练集顺序，每个迭代取batch_size个样本进行梯度下降
        batch_indexes = list(range(training_set_size))
        random.shuffle(batch_indexes)
        for iter in range(iteration_num):
            x_batch = X[batch_indexes[iter * Config.batch_size : (iter+1) * Config.batch_size]]
            y_batch = y[batch_indexes[iter * Config.batch_size : (iter+1) * Config.batch_size]]
            # 前向传播
            outputs, a = forward(x_batch, model)
            # 反向传播求梯度
            dW, db = bp(outputs, y_batch, a, model)
            # 梯度下降优化
            for layer_index in range(num_hidden_layers):
                W[layer_index] += -Config.learning_rate * dW[layer_index]
                b[layer_index] += -Config.learning_rate * db[layer_index]
            model["W"] = W
            model["b"] = b

        if epoch % 10 == 0:
            loss = calculate_loss(model, X, y)
            print("Loss @ epoch %i: %f" % (epoch, loss))
            capture_chart(X, y, model, epoch, loss)
    return model

def main():
    X, y = generate_data()
    layer_dims = [2, 5, 5, 5, 2]
    model = init_model(layer_dims, "tanh")
    train(X, y, model, 1000)
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=True, repeat_delay=0)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
