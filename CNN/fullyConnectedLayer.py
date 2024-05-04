import numpy as np

class fullyConnectedLayer:
    def __init__(self, input_size, output_size): # takes in input and output sizes
        self.input_size = input_size # a list
        self.output_size = output_size # the number of classes (2)
        self.weights = np.random.randn(input_size, output_size) 
        self.org_weights = self.weights  # keep track of the original randomized weights
        self.biases = np.zeros(output_size) # intialize biases to 0 (a 1D array with 2 elements)
        
        ## Testing weights and biases
        # self.weights = np.ones((input_size, output_size)) 
        # self.biases = np.ones(output_size) # intialize biases to 0 (a 1D array with 2 elements)
        
    # Forward propagation of the fully connected layer, input is a list. y = Wx + b
    def forward(self, input):
        self.input = np.array(input)  # turn the input list into a 1D numpy array
        output = np.dot(self.input, self.weights) + self.biases # output = xw + b 
        self.forward_output = output/self.output_size # record the last output calculated + normalization
        return self.forward_output
        
    # Backward propagation of the fully connected layer
    # learningRate: scalar controls how much the weights are adjusted in the gradient direction
    # dLdy = passed down partial derivatives, a list of partial derivatives of the cost w.r.t the output of the layer (y). These derivatives come from the softmax layer
    def backward(self, learning_rate, dL_dy):
        dL_dy = np.array(dL_dy)
        # get gradient of weights. dL/dW = dL/dy * dy/dW = dL/dy * x                                   
        gradient_of_weights = np.outer(self.input, dL_dy)
        # update the weights
        self.weights = self.weights - learning_rate * gradient_of_weights
        # get gradient of bias. dL/db = dL/dy * dy/db = dL/dy * 1 = dL/dy                               
        gradient_of_bias = dL_dy
        # update the biases
        self.biases = self.biases - learning_rate * gradient_of_bias 
        # Calculate gradients w.r.t the input of this layer (dL/dx = dL/dy * dy/dx) 
        self.backward_output = np.dot(dL_dy, self.weights.T)
        # return dL/dx the partial derivatives of the cost w.r.t the input of this layer  
        return self.backward_output 
    
    def saveWeights(self, file):
        with open(file, 'a') as file_out:
            file_out.write(f'fullyConnectedLayer, input_size, {self.input_size}, output_size, {self.output_size}, ')
            file_out.write('weights,')
            for c in range(self.output_size):
                for r in range(self.input_size):
                    file_out.write(f' {self.weights[r][c]}')
                file_out.write(',')
        
            file_out.write(' biases,')
            for i in range(self.output_size):
                file_out.write(f' {self.biases[i]}')
            file_out.write('\n')

    def loadWeights(self, input_size, output_size, weights, biases):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = weights
        self.biases = biases
            
    def printWeights(self):
        print('FCL Weights:')
        print(self.weights)

    def printBiasis(self):
        print('FCL Biases:')
        print(self.biases)