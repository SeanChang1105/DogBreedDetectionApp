from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import math
import convolution
import relu
import maxPool
import fullyConnectedLayer
import softMax
import layerNorm
import os
import sys


class model2:
    def __init__(self) -> None:
        self.convolutionLayer1=convolution.ConvolutionLayer()
        self.convolutionLayer2=convolution.ConvolutionLayer()
        self.convolutionLayer3=convolution.ConvolutionLayer()
        self.convolutionLayer4=convolution.ConvolutionLayer()
        self.normLayer=layerNorm.layerNorm()
        self.reluLayer=relu.Relu()
        self.maxPoolLayer=maxPool.MaxPool2D()
        self.fullyConnectedLayer=fullyConnectedLayer.fullyConnectedLayer(1024*4,2)
        self.softMaxLayer=softMax.SoftMax()

    def forward(self,img):
        cov_out1=self.convolutionLayer1.forward(img)
        cov_out2=self.convolutionLayer2.forward(img)
        cov_out3=self.convolutionLayer3.forward(img)
        cov_out4=self.convolutionLayer4.forward(img)
        norm_out1=self.normLayer.forward(cov_out1)
        norm_out2=self.normLayer.forward(cov_out2)
        norm_out3=self.normLayer.forward(cov_out3)
        norm_out4=self.normLayer.forward(cov_out4)
        relu_out1=self.reluLayer.forward(norm_out1)
        relu_out2=self.reluLayer.forward(norm_out2)
        relu_out3=self.reluLayer.forward(norm_out3)
        relu_out4=self.reluLayer.forward(norm_out4)
        pool_out1=self.maxPoolLayer.forward(relu_out1)
        pool_out2=self.maxPoolLayer.forward(relu_out2)
        pool_out3=self.maxPoolLayer.forward(relu_out3)
        pool_out4=self.maxPoolLayer.forward(relu_out4)
        full_input=np.concatenate((pool_out1.flatten(),pool_out2.flatten(),pool_out3.flatten(),pool_out4.flatten()))
        full_out=self.fullyConnectedLayer.forward(full_input)
        soft_out=self.softMaxLayer.forward(full_out)
        return soft_out

    def backward(self,loss,yTrue, learning_rate):
        soft_gradient=self.softMaxLayer.backward(yTrue)
        full_gradient=self.fullyConnectedLayer.backward(learning_rate,soft_gradient)
        pool_input=full_gradient.reshape(4,32,32)

        pool_gradient1=self.maxPoolLayer.backward(pool_input[0])
        relu_gradient1=self.reluLayer.backward(pool_gradient1)
        norm_gradient1=self.normLayer.backward(relu_gradient1)
        cov_gradient1=self.convolutionLayer1.backward(norm_gradient1,learning_rate)

        pool_gradient2=self.maxPoolLayer.backward(pool_input[1])
        relu_gradient2=self.reluLayer.backward(pool_gradient2)
        norm_gradient2=self.normLayer.backward(relu_gradient2)
        cov_gradient2=self.convolutionLayer2.backward(norm_gradient2,learning_rate)

        pool_gradient3=self.maxPoolLayer.backward(pool_input[2])
        relu_gradient3=self.reluLayer.backward(pool_gradient3)
        norm_gradient3=self.normLayer.backward(relu_gradient3)
        cov_gradient3=self.convolutionLayer3.backward(norm_gradient3,learning_rate)

        pool_gradient4=self.maxPoolLayer.backward(pool_input[3])
        relu_gradient4=self.reluLayer.backward(pool_gradient4)
        norm_gradient4=self.normLayer.backward(relu_gradient4)
        cov_gradient4=self.convolutionLayer4.backward(norm_gradient4,learning_rate)

    def saveModel(self, file):
        with open(file, 'w') as f:
            pass
        self.convolutionLayer.saveWeights(file)
        self.fullyConnectedLayer.saveWeights(file)

    def loadModel(self, file):
        with open(file, 'r') as f:
            for line in f.readlines():
                words = line.split(', ')
                if words[0] == 'Convolution':
                    size = int(words[1])
                    weights_arr = np.zeros((size, size))

                    for r in range(size):
                        weights_words = words[3+r].split()
                        for c in range(size):
                            weights_arr[r][c] = float(weights_words[c])

                    self.convolutionLayer.loadWeights(size, weights_arr)
                elif words[0] == 'fullyConnectedLayer':
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
                    self.fullyConnectedLayer.loadWeights(input_size, output_size, weights_arr, bias_arr)


class model3:
    def __init__(self) -> None:
        self.convolutionLayer1=convolution.ConvolutionLayer() #64x64x4
        self.convolutionLayer2=convolution.ConvolutionLayer()
        self.convolutionLayer3=convolution.ConvolutionLayer()
        self.convolutionLayer4=convolution.ConvolutionLayer()
        self.normLayer=layerNorm.layerNorm()
        self.reluLayer=relu.Relu()
        self.maxPoolLayer=maxPool.MaxPool2D() #32x32x4

        self.convolutionLayer1_1=convolution.ConvolutionLayer() #32x32x8
        self.convolutionLayer1_2=convolution.ConvolutionLayer()
        self.convolutionLayer2_1=convolution.ConvolutionLayer()
        self.convolutionLayer2_2=convolution.ConvolutionLayer()
        self.convolutionLayer3_1=convolution.ConvolutionLayer()
        self.convolutionLayer3_2=convolution.ConvolutionLayer()
        self.convolutionLayer4_1=convolution.ConvolutionLayer()
        self.convolutionLayer4_2=convolution.ConvolutionLayer()
        self.normLayer_2=layerNorm.layerNorm()
        self.reluLayer_2=relu.Relu()
        self.maxPoolLayer_2=maxPool.MaxPool2D() #16x16x8
        self.fullyConnectedLayer=fullyConnectedLayer.fullyConnectedLayer(16*16*8,2)
        self.softMaxLayer=softMax.SoftMax()

    def forward(self,img):
        cov_out1=self.convolutionLayer1.forward(img)
        cov_out2=self.convolutionLayer2.forward(img)
        cov_out3=self.convolutionLayer3.forward(img)
        cov_out4=self.convolutionLayer4.forward(img)
        norm_out1=self.normLayer.forward(cov_out1)
        norm_out2=self.normLayer.forward(cov_out2)
        norm_out3=self.normLayer.forward(cov_out3)
        norm_out4=self.normLayer.forward(cov_out4)
        relu_out1=self.reluLayer.forward(norm_out1)
        relu_out2=self.reluLayer.forward(norm_out2)
        relu_out3=self.reluLayer.forward(norm_out3)
        relu_out4=self.reluLayer.forward(norm_out4)
        pool_out1=self.maxPoolLayer.forward(relu_out1)
        pool_out2=self.maxPoolLayer.forward(relu_out2)
        pool_out3=self.maxPoolLayer.forward(relu_out3)
        pool_out4=self.maxPoolLayer.forward(relu_out4)

        cov_out1_1=self.convolutionLayer1_1.forward(pool_out1)
        cov_out1_2=self.convolutionLayer1_2.forward(pool_out1)
        cov_out2_1=self.convolutionLayer2_1.forward(pool_out2)
        cov_out2_2=self.convolutionLayer2_2.forward(pool_out2)
        cov_out3_1=self.convolutionLayer3_1.forward(pool_out3)
        cov_out3_2=self.convolutionLayer3_2.forward(pool_out3)
        cov_out4_1=self.convolutionLayer4_1.forward(pool_out4)
        cov_out4_2=self.convolutionLayer4_2.forward(pool_out4)
        norm_out1_1=self.normLayer_2.forward(cov_out1_1)
        norm_out1_2=self.normLayer_2.forward(cov_out1_2)
        norm_out2_1=self.normLayer_2.forward(cov_out2_1)
        norm_out2_2=self.normLayer_2.forward(cov_out2_2)
        norm_out3_1=self.normLayer_2.forward(cov_out3_1)
        norm_out3_2=self.normLayer_2.forward(cov_out3_2)
        norm_out4_1=self.normLayer_2.forward(cov_out4_1)
        norm_out4_2=self.normLayer_2.forward(cov_out4_2)
        relu_out1_1=self.reluLayer_2.forward(norm_out1_1)
        relu_out1_2=self.reluLayer_2.forward(norm_out1_2)
        relu_out2_1=self.reluLayer_2.forward(norm_out2_1)
        relu_out2_2=self.reluLayer_2.forward(norm_out2_2)
        relu_out3_1=self.reluLayer_2.forward(norm_out3_1)
        relu_out3_2=self.reluLayer_2.forward(norm_out3_2)
        relu_out4_1=self.reluLayer_2.forward(norm_out4_1)
        relu_out4_2=self.reluLayer_2.forward(norm_out4_2)
        pool_out1_1=self.maxPoolLayer_2.forward(relu_out1_1)
        pool_out1_2=self.maxPoolLayer_2.forward(relu_out1_2)
        pool_out2_1=self.maxPoolLayer_2.forward(relu_out2_1)
        pool_out2_2=self.maxPoolLayer_2.forward(relu_out2_2)
        pool_out3_1=self.maxPoolLayer_2.forward(relu_out3_1)
        pool_out3_2=self.maxPoolLayer_2.forward(relu_out3_2)
        pool_out4_1=self.maxPoolLayer_2.forward(relu_out4_1)
        pool_out4_2=self.maxPoolLayer_2.forward(relu_out4_2)

        full_input=np.concatenate((pool_out1_1.flatten(),pool_out1_2.flatten(),pool_out2_1.flatten(),pool_out2_2.flatten(),
                                   pool_out3_1.flatten(),pool_out3_2.flatten(),pool_out4_1.flatten(),pool_out4_2.flatten()))
        full_out=self.fullyConnectedLayer.forward(full_input)
        soft_out=self.softMaxLayer.forward(full_out)
        return soft_out

    def backward(self,loss,yTrue, learning_rate):
        soft_gradient=self.softMaxLayer.backward(yTrue)
        full_gradient=self.fullyConnectedLayer.backward(learning_rate,soft_gradient)
        pool_2_input=full_gradient.reshape(8,16,16)

        pool_gradient1_1=self.maxPoolLayer_2.backward(pool_2_input[0])
        relu_gradient1_1=self.reluLayer_2.backward(pool_gradient1_1)
        norm_gradient1_1=self.normLayer_2.backward(relu_gradient1_1)
        cov_gradient1_1=self.convolutionLayer1_1.backward(norm_gradient1_1,learning_rate)

        pool_gradient1_2=self.maxPoolLayer_2.backward(pool_2_input[1])
        relu_gradient1_2=self.reluLayer_2.backward(pool_gradient1_2)
        norm_gradient1_2=self.normLayer_2.backward(relu_gradient1_2)
        cov_gradient1_2=self.convolutionLayer1_2.backward(norm_gradient1_2,learning_rate)

        pool_gradient2_1=self.maxPoolLayer_2.backward(pool_2_input[2])
        relu_gradient2_1=self.reluLayer_2.backward(pool_gradient2_1)
        norm_gradient2_1=self.normLayer_2.backward(relu_gradient2_1)
        cov_gradient2_1=self.convolutionLayer2_1.backward(norm_gradient2_1,learning_rate)

        pool_gradient2_2=self.maxPoolLayer_2.backward(pool_2_input[3])
        relu_gradient2_2=self.reluLayer_2.backward(pool_gradient2_2)
        norm_gradient2_2=self.normLayer_2.backward(relu_gradient2_2)
        cov_gradient2_2=self.convolutionLayer2_2.backward(norm_gradient2_2,learning_rate)

        pool_gradient3_1=self.maxPoolLayer_2.backward(pool_2_input[4])
        relu_gradient3_1=self.reluLayer_2.backward(pool_gradient3_1)
        norm_gradient3_1=self.normLayer_2.backward(relu_gradient3_1)
        cov_gradient3_1=self.convolutionLayer3_1.backward(norm_gradient3_1,learning_rate)

        pool_gradient3_2=self.maxPoolLayer_2.backward(pool_2_input[5])
        relu_gradient3_2=self.reluLayer_2.backward(pool_gradient3_2)
        norm_gradient3_2=self.normLayer_2.backward(relu_gradient3_2)
        cov_gradient3_2=self.convolutionLayer3_2.backward(norm_gradient3_2,learning_rate)

        pool_gradient4_1=self.maxPoolLayer_2.backward(pool_2_input[6])
        relu_gradient4_1=self.reluLayer_2.backward(pool_gradient4_1)
        norm_gradient4_1=self.normLayer_2.backward(relu_gradient4_1)
        cov_gradient4_1=self.convolutionLayer4_1.backward(norm_gradient4_1,learning_rate)

        pool_gradient4_2=self.maxPoolLayer_2.backward(pool_2_input[7])
        relu_gradient4_2=self.reluLayer_2.backward(pool_gradient4_2)
        norm_gradient4_2=self.normLayer_2.backward(relu_gradient4_2)
        cov_gradient4_2=self.convolutionLayer4_2.backward(norm_gradient4_2,learning_rate)

        cov_2_gradient_combined=np.concatenate((cov_gradient1_1+cov_gradient1_2,cov_gradient2_1+cov_gradient2_2,
                                                cov_gradient3_1+cov_gradient3_2,cov_gradient4_1+cov_gradient4_2))
        pool_input=cov_2_gradient_combined.reshape(4,32,32)

        pool_gradient1=self.maxPoolLayer.backward(pool_input[0])
        relu_gradient1=self.reluLayer.backward(pool_gradient1)
        norm_gradient1=self.normLayer.backward(relu_gradient1)
        cov_gradient1=self.convolutionLayer1.backward(norm_gradient1,learning_rate)

        pool_gradient2=self.maxPoolLayer.backward(pool_input[1])
        relu_gradient2=self.reluLayer.backward(pool_gradient2)
        norm_gradient2=self.normLayer.backward(relu_gradient2)
        cov_gradient2=self.convolutionLayer2.backward(norm_gradient2,learning_rate)

        pool_gradient3=self.maxPoolLayer.backward(pool_input[2])
        relu_gradient3=self.reluLayer.backward(pool_gradient3)
        norm_gradient3=self.normLayer.backward(relu_gradient3)
        cov_gradient3=self.convolutionLayer3.backward(norm_gradient3,learning_rate)

        pool_gradient4=self.maxPoolLayer.backward(pool_input[3])
        relu_gradient4=self.reluLayer.backward(pool_gradient4)
        norm_gradient4=self.normLayer.backward(relu_gradient4)
        cov_gradient4=self.convolutionLayer4.backward(norm_gradient4,learning_rate)


    def saveModel(self, file):
        with open(file, 'w') as f:
            pass
        self.convolutionLayer.saveWeights(file)
        self.fullyConnectedLayer.saveWeights(file)

    def loadModel(self, file):
        with open(file, 'r') as f:
            for line in f.readlines():
                words = line.split(', ')
                if words[0] == 'Convolution':
                    size = int(words[1])
                    weights_arr = np.zeros((size, size))

                    for r in range(size):
                        weights_words = words[3+r].split()
                        for c in range(size):
                            weights_arr[r][c] = float(weights_words[c])

                    self.convolutionLayer.loadWeights(size, weights_arr)
                elif words[0] == 'fullyConnectedLayer':
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
                    self.fullyConnectedLayer.loadWeights(input_size, output_size, weights_arr, bias_arr)