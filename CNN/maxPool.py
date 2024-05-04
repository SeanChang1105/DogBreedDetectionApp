import numpy as np

class MaxPool2D:
    def __init__(self, stride=2):
        self.size = 2
        self.stride = stride

    def region(self,image):
        height = int(len(image))
        width = int(len(image[0]))
        
        self.image = image
        
        for i in range(0, height - self.size + 1, self.stride):
            for j in range(0, width - self.size + 1, self.stride):
                patch = image[i : i+self.size, j : j+self.size]
                yield patch,i//self.stride,j//self.stride

    def forward(self, input):
        height, width = input.shape
        self.image = input

        out = np.zeros(((height-self.size) // self.stride + 1, (width-self.size) // self.stride + 1))
        for region,i,j in self.region(self.image):
            out[i,j] = np.amax(region,axis=(0,1))

        return out

    def backward(self, dL_dy):
        # Currently if there is a matrix [[0, 0], [0, 0]] then all four spots are the max
        #initialize other non max values to 0
        dL_dx = np.zeros(self.image.shape)
        
        for patch,i,j in self.region(self.image):
            height, width = patch.shape
            
            old_max = np.amax(patch,axis=(0,1))
            
            for k in range(height):
                for l in range(width):
                    if patch[k, l] == old_max:
                        #assigns the gradient
                        if np.isscalar(dL_dy):
                            # dL_dy is a single int
                            dL_dx[i * self.stride + k, j * self.stride + l] += dL_dy
                        else:
                            # dL_dy has the same shape as the image
                            dL_dx[i * self.stride + k, j * self.stride + l] += dL_dy[i, j]
         
        return dL_dx