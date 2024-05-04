import matplotlib.pyplot as plt
import numpy as np

# Generates the Gaussian filter kernel. size = Size of the kernel, sigma = standard deviation
def gaussianFilter(size: int, sigma: float):
    # Initialize kernel as an array of zeros
    kernel = np.zeros((size, size))
    
    # Calculate center of the kernel
    center = size // 2
    
    # Compute Gaussian function for each coordinate
    for x in range(size):
        for y in range(size):
            kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
            
    # Normalize the kernel
    return kernel / np.sum(kernel)


# Sobel Edge Detection (Grayscale, Gaussian Filter, Sobel Operator, Gradient of Magnitude)
input = '../Dataset/golden_retriever.jpg'
image = plt.imread(input)

# 1. Grayscale
#isolate image's RGB channel
r = image[:, :, 0]
g = image[:, :, 1]
b = image[:, :, 2]

#grayscale the image using BT.709 std
grayImg = 0.2126 * r + 0.7152 * g + 0.0722*b

# 2. Gaussian Filter
gaussianfilter = np.copy(grayImg)
height, width = grayImg.shape
kernel = gaussianFilter(3, 3)
for x in range(1, height-1):
    for y in range(1, width-1):
        gaussianfilter[x, y] = np.sum(grayImg[x-1:x+2, y-1:y+2] * kernel)
    
        
# 3. Sobel Operator
# Sobel kernel
horizontal = np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]])
vertical = np.array([[-1,-2,-1],
                     [0,0,0],
                     [1,2,1]])

horizontalGradient = np.zeros([height, width])
verticalGradient = np.zeros([height, width])


for x in range(1, height-1):
        for y in range(1, width-1):
            horizontalGradient[x, y] = np.sum(gaussianfilter[x-1:x+2, y-1:y+2] * horizontal)
            verticalGradient[x, y] = np.sum(gaussianfilter[x-1:x+2, y-1:y+2] * vertical)
            
magnitude = np.sqrt(horizontalGradient **2 + verticalGradient**2)


# Apply thresholding
threshold = 100
# if > threshold, 255 (white). else 0 (black)
final = np.where(magnitude > threshold, 255, 0)
plt.imshow(final, cmap='gray')
plt.title('Edge DetectionGaussian')
plt.imsave('edgeDetectionGaussian.jpg',final,cmap='gray')
plt.show()



