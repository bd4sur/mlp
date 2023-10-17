# forked from https://github.com/dennybritz/nn-from-scratch
# ref https://dennybritz.com/posts/wildml/implementing-a-neural-network-from-scratch/
# BD4SUR 2023.10

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength

def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(500, noise=0.30)
    return X, y

def visualize(X, y, model):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x:predict(model,x), X, y)
    plt.title("Logistic Regression")

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def forward(x, W, b):
    num_hidden_layers = len(W)
    a = [0] * (num_hidden_layers + 1)
    z = [0] * (num_hidden_layers + 1)
    a[0] = x
    for i in range(num_hidden_layers):
        Wi = W[i]
        bi = b[i]
        z[i+1] = a[i].dot(Wi) + bi
        a[i+1] = np.tanh(z[i+1])
    # softmax
    exp_scores = np.exp(z[num_hidden_layers])
    y_predict = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return (y_predict, a)

def bp(y_predict, y, W, b, a, num_examples):
    num_hidden_layers = len(W)
    delta = y_predict
    delta[range(num_examples), y] -= 1
    dW = [0] * num_hidden_layers
    db = [0] * num_hidden_layers
    for i in range(num_hidden_layers):
        i = num_hidden_layers - i - 1
        dW[i] = (a[i].T).dot(delta) + Config.reg_lambda * W[i] # L2-regularization
        if i == num_hidden_layers - 1:
            db[i] = np.sum(delta, axis=0, keepdims=True)
        else:
            db[i] = np.sum(delta, axis=0)
        delta = delta.dot(W[i].T) * (1 - np.power(a[i], 2))
    return (dW, db)

def calculate_loss(model, X, y):
    num_examples = len(X)  # training set size
    W, b = model['W'], model['b']
    # Forward propagation to calculate our predictions
    y_predict = forward(X, W, b)[0]
    # Calculating the loss
    corect_logprobs = -np.log(y_predict[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add L2 regulatization term to loss (optional)
    sum = 0
    for i in range(len(W)):
        sum += np.sum(np.square(W[i]))
    data_loss += Config.reg_lambda / 2 * sum
    return 1. / num_examples * data_loss

def predict(model, x):
    W, b = model['W'], model['b']
    # Forward propagation
    y_predict = forward(x, W, b)[0]
    return np.argmax(y_predict, axis=1)

# layer_dims = [input_layer_dim, hidden1, hidden2, ... , output_layer_dim]
def build_model(X, y, layer_dims, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = len(X)

    num_hidden_layers = len(layer_dims) - 1
    np.random.seed(0)

    W = [0] * num_hidden_layers
    b = [0] * num_hidden_layers
    for layer_index in range(num_hidden_layers):
        W[layer_index] = np.random.randn(layer_dims[layer_index], layer_dims[layer_index + 1]) / np.sqrt(layer_dims[layer_index])
        b[layer_index] = np.zeros((1, layer_dims[layer_index + 1]))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        y_predict, a = forward(X, W, b)

        # Backpropagation
        dW, db = bp(y_predict, y, W, b, a, num_examples)

        # Gradient descent parameter update
        for layer_index in range(num_hidden_layers):
            W[layer_index] += -Config.epsilon * dW[layer_index]
            b[layer_index] += -Config.epsilon * db[layer_index]

        # Assign new parameters to the model
        model = {'W': W, 'b': b}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))

    return model


def classify(X, y):
    # clf = linear_model.LogisticRegressionCV()
    # clf.fit(X, y)
    # return clf

    pass


def main():
    X, y = generate_data()
    model = build_model(X, y, [Config.nn_input_dim, 5, 4, 3, Config.nn_output_dim], print_loss=True)
    visualize(X, y, model)


if __name__ == "__main__":
    main()
