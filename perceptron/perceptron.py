import itertools
import numpy as np
import pandas as pd
from util import relu, sig, step_func, binary_threshold
from util import load_data

class Perceptron(object):

  def __init__(self, activation_function, n_inputs, weights, epochs=100, learning_rate=0.01):
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.activation_function = activation_function
    self.n_inputs = n_inputs
    self.weights = weights

  def predict(self, inputs):
    pred = np.dot(self.weights.T, inputs)
    return binary_threshold(self.activation_function(pred))

  def train(self, training_inputs, labels):
    
    for _ in np.arange(0, self.epochs):
      for (input, label) in zip(training_inputs.to_numpy(), labels.to_numpy()):
        pred = self.predict(input)
        error = label - pred
        self.weights = self.weights + input*error*self.learning_rate
  
  def score(self, inputs, labels):
    tp_tn = 0

    for (input, label) in zip(inputs.to_numpy(), labels.to_numpy()):
      pred = self.predict(input)

      if pred == label:
        tp_tn += 1
    
    return tp_tn / len(inputs.to_numpy())

if __name__ == "__main__":
    train_X, test_X, train_Y, test_Y = load_data()
    train_X["bias"] = 1
    test_X["bias"] = 1

    weights = np.random.randn(train_X.shape[1]) / np.sqrt(train_X.shape[1])
    learning_rates = [0.01, 0.1]
    activation_functions = [step_func, sig, relu]
    epochs = [100, 500, 1000]

    hyper_p = itertools.product(learning_rates, activation_functions, epochs)
    outputs = []
    for (learning_rate, activation_function, epoch) in hyper_p:
        perceptron = Perceptron(learning_rate=learning_rate,
                                activation_function=activation_function,
                                weights=weights,
                                n_inputs=train_X.shape[1],
                                epochs=epoch)

        perceptron.train(train_X, train_Y)
        acc = perceptron.score(test_X, test_Y)
  
        outputs.append([activation_function.__name__, learning_rate, epoch, acc])

    output_frame = pd.DataFrame(outputs, columns=["activation_function", "learning_rate", "epochs", "accuracy"])
    
    print(output_frame.to_string(index=False))
    output_frame.to_csv("metrics_output.csv", index=False)
    