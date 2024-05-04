import numpy as np

class ConvolutionLayer:
    def __init__(self):
        self.size_filter = 5
        # Figure out how much to pad the image by
        self.pad_size = int((self.size_filter-1)/2)
        self.filter = np.random.randn(self.size_filter, self.size_filter)

        ## Testing filter
        # self.filter = np.ones((self.size_filter, self.size_filter))

    def forward(self, image):
        self.image = image
        #get the shape of the image and filter
        image_shape = image.shape[0]
        # Pad the image
        self.padded_image = padded_image = np.pad(image, self.pad_size, mode='constant', constant_values=0)

        image_height, image_width = image.shape
        kernal_size = len(self.filter)

        # Create a feature map
        feature_map = np.zeros_like(image)
        
        # Perform Convolution
        for i in range(self.pad_size, image_height + self.pad_size):
            for j in range(self.pad_size, image_width + self.pad_size):
                neighborhood = padded_image[i - self.pad_size:i + self.pad_size + 1, j - self.pad_size:j + self.pad_size + 1]
                feature_map[i - self.pad_size, j - self.pad_size] = np.sum(neighborhood * self.filter)
        
        return feature_map
    
    def backward(self, dL_dy, learning_rate):
        img_shape = self.image.shape[0]
        pad_img_shape = self.padded_image.shape[0]
        filter_shape = self.filter.shape[0]
        output_shape = img_shape - filter_shape + 1
        dL_dfilter = np.zeros((filter_shape, filter_shape))
        dL_dout = np.zeros((pad_img_shape, pad_img_shape))

        # Getting dL_dw and dL_dout
        for r in range(img_shape):
            for c in range(img_shape): 
                dL_dfilter += dL_dy[r][c] * self.padded_image[r:r+self.size_filter, c:c+self.size_filter]
                dL_dout[r:r+self.size_filter, c:c+self.size_filter] += dL_dy[r][c] * self.filter

        # Update the weights in the filter
        self.filter = self.filter - learning_rate * dL_dfilter

        # returned dL_dx the loss with respect to the input matrix
        return dL_dout[self.pad_size:pad_img_shape - self.pad_size, self.pad_size:pad_img_shape - self.pad_size]
    
    def saveWeights(self, file):
        with open(file, 'a') as file_out:
            file_out.write(f'Convolution, {self.size_filter}, ')
            file_out.write('weights,')
            for i in range(self.size_filter):
                for j in range(self.size_filter):
                    file_out.write(f' {self.filter[i][j]}')
                if i < self.size_filter - 1:
                    file_out.write(',')
            file_out.write('\n')

    def loadWeights(self, size, filter_weights):
        self.size_filter = size
        self.filter = filter_weights

    def printWeights(self):
        print("Convolution Filter Weights:")
        print(self.filter)