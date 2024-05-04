import numpy as np
import matplotlib.pyplot as plt
import os


def grayscale(img: np.array):
   #isolate image's RGB channel
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    #grayscale the image using BT.709 std
    gray_img = 0.2126 * r + 0.7152 * g + 0.0722*b
    return gray_img

# old resize method
def old_resize(img:np.array, new_size: int):
    # Create a black (pixel val 0) img with the new size
    new_img = np.zeros((new_size, new_size), dtype=np.uint8)
    height, width = img.shape
    # Calculate the new dimensions based on the aspect ratio
    if height < width:
        new_height = int(new_size*(height/width))
        new_width = new_size
    elif height == width:
        new_height = new_size
        new_width = new_size
    else:
        new_height = new_size
        new_width = int(new_size*(width/height))
    
    # Calculate padding on x and y dimensions
    x_pad = (new_size - new_height) // 2
    y_pad = (new_size - new_width) // 2

    # Resize
    for x in range(new_height):
        for y in range(new_width):
            # Calculate the corresponding pixel position in the original image
            x_original = int(x*(height/new_height))
            y_original = int(y*(width/new_width))
            new_img[x + x_pad, y + y_pad] = img[x_original, y_original]
    return new_img

# inter area method
def resize(image:np.array, new_size: 'tuple[int, int]'):
    # Compute scaling factors
    scale_x = image.shape[1] / new_size[1]
    scale_y = image.shape[0] / new_size[0]
    
    resized_image = np.zeros((new_size[0], new_size[1]), dtype=np.uint8)
    
    for y in range(new_size[0]):
        for x in range(new_size[1]):
            # Calculate the corresponding pixel position in the original image
            x_original = int(x * scale_x)
            y_original = int(y * scale_y)
            
            # If within the bounds
            if x_original < image.shape[1] and y_original < image.shape[0]:
                # Assign pixel value from input img to output img
                resized_image[y, x] = image[y_original, x_original]
    
    return resized_image


# resize multiple images all at once
def auto_resize():
    original_img_dir = 'DogDataset'
    resized_img_dir = 'ResizedImage'
    # list all subdir in the dir
    subdir=os.listdir(original_img_dir)
    for dir in subdir:
        # ignore .DS_Store file for mac
        if dir =='.DS_Store':
            continue
        train_test_dir=os.path.join(original_img_dir,dir)
        resized_train_test_dir=os.path.join(resized_img_dir,dir)
        # list all the categories
        dogCategories = os.listdir(train_test_dir)
        for category in dogCategories:
            # ignore .DS_Store file for mac
            if category=='.DS_Store':
                continue
            # construct paths 
            ori_category_path = os.path.join(train_test_dir, category)
            resized_category_path = os.path.join(resized_train_test_dir, category)
            
            # create the resized image dir if dne
            os.makedirs(resized_category_path, exist_ok=True)
            
            # list all images
            images = os.listdir(ori_category_path)
            for image in images:
                if image.endswith('.jpg'):
                    img = plt.imread(os.path.join(ori_category_path, image)) # constructs the full path to the image
                    # grayscale
                    grayscaled_img = grayscale(img)
                    # resize
                    resize_img = resize(grayscaled_img, (64,64))
                    
                    # save grayscale image
                    #grayscale_path = os.path.join(img_dir, 'grayscale_' + image)
                    #plt.imsave(grayscale_path, grayscaled_img, cmap = 'gray')
                
                    # save resized image
                    resize_path = os.path.join(resized_category_path, 'resize_' + image)
                    plt.imsave(resize_path, resize_img, cmap = 'gray')