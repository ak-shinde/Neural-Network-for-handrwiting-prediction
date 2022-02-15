import numpy as np
import pandas as pd
import sys

LEARNING_RATE = 0.25

MAX_ITERATIONS = 1000

INPUT_NEURONS = 784
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 64
OUTPUT_NEURONS = 10

MAX_PIXEL_RGB = 255
OUTPUT_FILE_NAME = 'test_predictions.csv'

class NeuralNetwork:
    def __init__(self):
        train_file = sys.argv[1]
        train_label_file = sys.argv[2]
        test_file = sys.argv[3]

        self.X_train = np.array(pd.read_csv(train_file, header=None))
        self.Y_train = np.array(pd.read_csv(train_label_file, header=None)).flatten()
        self.X_test = np.array(pd.read_csv(test_file, header=None))

        self.m, self.n = self.X_train.shape

        self.X_train = self.X_train.T / 255.
        self.X_test = self.X_test.T / 255.
    
    class Layer:
        def __init__(self, features, neurons):
            self.W = np.random.rand(neurons, features) - 0.5
            self.b = np.random.rand(neurons, 1) - 0.5
        
        def forward(self, X):
            self.Z = self.W.dot(X) + self.b
        
        def backward_output(self, m, output_activation, prev_activation, Y):
            one_hot_Y = self.one_hot(Y)
            self.dZ = output_activation.A - one_hot_Y
            self.dW = 1 / m * self.dZ.dot(prev_activation.A.T)
            self.db = 1 / m * np.sum(self.dZ)
        
        def backward(self, m, next_layer, input, activation):
            self.dZ = next_layer.W.T.dot(next_layer.dZ) * activation.dA
            self.dW = 1 / m * self.dZ.dot(input.T)
            self.db = 1 / m * np.sum(self.dZ)

        def update_parameters(self, alpha):
            self.W = self.W - alpha * self.dW
            self.b = self.b - alpha * self.db
        
        def one_hot(self, Y):
            one_hot_Y = np.zeros((Y.size, OUTPUT_NEURONS))
            one_hot_Y[np.arange(Y.size), Y] = 1
            one_hot_Y = one_hot_Y.T
            return one_hot_Y
    
    class Activation:
        class Sigmoid:
            def forward(self, Z):
                self.A = 1 / (1 + np.exp(-Z))
            
            def backward(self, Z):
                Z = 1 / (1 + np.exp(-Z))
                self.dA = Z * (1 - Z)
        
        class Softmax:
            def forward(self, Z):
                self.A = np.exp(Z) / sum(np.exp(Z))

            def backward(self):
                pass
        
            def get_prediction(self):
                return np.argmax(self.A, 0)
    
    
    def forward(self, layer1: Layer, layer2: Layer, output_layer: Layer, activation1: Activation.Sigmoid, activation2: Activation.Sigmoid, output_activation: Activation.Softmax, X):
        layer1.forward(X)
        activation1.forward(layer1.Z)
        layer2.forward(activation1.A)
        activation2.forward(layer2.Z)
        output_layer.forward(activation2.A)
        output_activation.forward(output_layer.Z)
    
    def backward(self, layer1: Layer, layer2: Layer, output_layer: Layer, activation1: Activation.Sigmoid, activation2: Activation.Sigmoid, output_activation: Activation.Softmax, X, Y):
        output_layer.backward_output(self.m, output_activation, activation2, Y)
        activation2.backward(layer2.Z)
        layer2.backward(self.m, output_layer, activation1.A, activation2)
        activation1.backward(layer1.Z)
        layer1.backward(self.m, layer2, X, activation1)
    
    def update_parameters(self, layer1: Layer, layer2: Layer, output_layer: Layer, alpha):
        layer1.update_parameters(alpha)
        layer2.update_parameters(alpha)
        output_layer.update_parameters(alpha)

    def get_accuracy(self, prediction):
        print(prediction, self.Y_train)
        return np.sum(prediction == self.Y_train) / self.Y_train.size
    
    def make_predictions(self, layer1: Layer, layer2: Layer, output_layer: Layer, activation1: Activation.Sigmoid, activation2: Activation.Sigmoid, output_activation: Activation.Softmax, X):
        self.forward(layer1, layer2, output_layer, activation1, activation2, output_activation, X)
        prediction = output_activation.get_prediction()
        print(NN.get_accuracy(prediction))
        np.savetxt(OUTPUT_FILE_NAME, prediction, fmt='%d', newline='\n')

if __name__ == '__main__':
    NN = NeuralNetwork()
    layer1 = NN.Layer(INPUT_NEURONS, HIDDEN_LAYER_1)
    layer2 = NN.Layer(HIDDEN_LAYER_1, HIDDEN_LAYER_2)
    output_layer = NN.Layer(HIDDEN_LAYER_2, OUTPUT_NEURONS)

    activation1 = NN.Activation().Sigmoid()
    activation2 = NN.Activation().Sigmoid()
    output_activation = NN.Activation().Softmax()

    for epoch in range(MAX_ITERATIONS):
        NN.forward(layer1, layer2, output_layer, activation1, activation2, output_activation, NN.X_train)
        NN.backward(layer1, layer2, output_layer, activation1, activation2, output_activation, NN.X_train, NN.Y_train)
        NN.update_parameters(layer1, layer2, output_layer, LEARNING_RATE)
        if (epoch % 100 == 0):
            prediction = output_activation.get_prediction()
            print(NN.get_accuracy(prediction))
    
    NN.make_predictions(layer1, layer2, output_layer, activation1, activation2, output_activation, NN.X_test)
