import numpy as np

class Relu:
    def forward(self,image):
        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        # Get the shape of the input dataset and create an output array with the same shape
        relu_out = np.zeros((self.height, self.width))
        # For each layer in the input dataset, perform relu for each pixel of the image
        for i in np.arange(0, self.height):
           for j in np.arange(0, self.width):
                # ReLU = max(x, 0)
                relu_out[i, j] = np.max([self.image[i, j], 0])
        return relu_out
    
    def backward(self, dL_dy):
        # dLdy: the gradient of the loss function wrt the output of the layer, passed from maxpool
        dy_dx = np.array(self.image)
        dy_dx[self.image <= 0] = 0
        dy_dx[self.image > 0]  = 1
        #gradients w.r.t the input of this layer dL/dx
        # dL/dx = dL/dy * dy/dx = dL/dy * dy/dx 
        dL_dx= dL_dy * dy_dx
        return dL_dx