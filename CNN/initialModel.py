from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import math
import convolution
import relu
import maxPool
import fullyConnectedLayer
import softMax
import os


class model1:
    def __init__(self) -> None:
        self.convolutionLayer=convolution.ConvolutionLayer()
        self.reluLayer=relu.Relu()
        self.maxPoolLayer=maxPool.MaxPool2D()
        self.fullyConnectedLayer=fullyConnectedLayer.fullyConnectedLayer(1024,2)
        self.softMaxLayer=softMax.SoftMax()

        self.layer_array = [self.convolutionLayer, self.fullyConnectedLayer]

    def forward(self,img):
        cov_out=self.convolutionLayer.forward(img)
        relu_out=self.reluLayer.forward(cov_out)
        pool_out=self.maxPoolLayer.forward(relu_out)
        full_input=pool_out.flatten()
        full_out=self.fullyConnectedLayer.forward(full_input)
        soft_out=self.softMaxLayer.forward(full_out)
        return soft_out

    def backward(self,loss,yTrue, learning_rate):
        soft_gradient=self.softMaxLayer.backward(yTrue)
        full_gradient=self.fullyConnectedLayer.backward(learning_rate,soft_gradient)
        pool_input=full_gradient.reshape(32,32)
        pool_gradient=self.maxPoolLayer.backward(pool_input)
        relu_gradient=self.reluLayer.backward(pool_gradient)
        cov_gradient=self.convolutionLayer.backward(relu_gradient,learning_rate)

    def saveModel(self, file):
        with open(file, 'w') as f:
            pass
        for layer in self.layer_array:
            layer.saveWeights(file)

    def loadModel(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()

            if len(lines) != len(self.layer_array):
                print('Lengths Dont match.')
                return

            for step, line in enumerate(lines, start=0):
                words = line.split(', ')
                if words[0] == 'Convolution' and isinstance(self.layer_array[step], convolution.ConvolutionLayer):
                    size = int(words[1])
                    weights_arr = np.zeros((size, size))

                    for r in range(size):
                        weights_words = words[3+r].split()
                        for c in range(size):
                            weights_arr[r][c] = float(weights_words[c])

                    self.layer_array[step].loadWeights(size, weights_arr)
                elif words[0] == 'fullyConnectedLayer' and isinstance(self.layer_array[step], fullyConnectedLayer.fullyConnectedLayer):
                    input_size = int(words[2])
                    output_size = int(words[4])
                    weights_arr = np.zeros((input_size, output_size))
                    bias_arr = np.zeros(output_size)

                    for i in range(output_size):
                        weights_words = words[6+i].split()
                        for j in range(input_size):
                            weights_arr[j][i] = float(weights_words[j])

                    biases = words[-1].split()
                    for i in range(output_size):
                        bias_arr[i] = float(biases[i])
                    self.layer_array[step].loadWeights(input_size, output_size, weights_arr, bias_arr)
                else:
                    print(f'Word: {words[0]}, {self.layer_array[step]}')
                    print('Input and Model do not match')
                    return