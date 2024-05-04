import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# load training data and ground truth into list
def loadTrainingData(shuffle:bool):
    preprocessing.auto_resize()
    resized_img_dir = 'ResizedImage/training_data'
    # list all subdir in the dir
    dogCategories = os.listdir(resized_img_dir)
    # The list of arrays that is being output
    data_list=[]
    # The list of ground truths
    ground_truth_list=[]
    # golden_retriever=0, siberian_husky=1
    
    if shuffle: #shuffle the order
        path_list=[]
        for category in dogCategories:
            # ignore .DS_Store file for mac
            if category=='.DS_Store':
                continue
            # construct paths 
            resized_category_path = os.path.join(resized_img_dir, category)
            
            # list all images
            images = os.listdir(resized_category_path)
            for image_path in images:
                if image_path.endswith('.jpg'):
                    path_list.append(image_path)
        # shuffle the dataset
        random.shuffle(path_list)
        for path in path_list:
            if "husky" in path:# data is husky
                img = plt.imread(os.path.join("ResizedImage/training_data/siberian_husky/", path)) # constructs the full path to the image
                data_list.append(img[:,:,0]/255)
                ground_truth_list.append(np.array([0,1])) # data is husky
            else:
                img = plt.imread(os.path.join("ResizedImage/training_data/golden_retriever/", path)) # constructs the full path to the image
                data_list.append(img[:,:,0]/255)
                ground_truth_list.append(np.array([1,0])) # data is golden retreiver
            
    else: # store img in order
        ground_truth_idx=np.array([1,0])
        for category in dogCategories:
            # ignore .DS_Store file for mac
            if category=='.DS_Store':
                continue
            # construct paths 
            resized_category_path = os.path.join(resized_img_dir, category)
            
            # list all images
            images = os.listdir(resized_category_path)
            for image_path in images:
                if image_path.endswith('.jpg'):
                    img = plt.imread(os.path.join(resized_category_path, image_path)) # constructs the full path to the image
                    data_list.append(img[:,:,0]/255)
                    ground_truth_list.append(ground_truth_idx)
            # update the idx for next iteration
            ground_truth_idx=np.array([0,1])

    return data_list,ground_truth_list


# load testing data and ground truth
def loadTestingData(shuffle:bool):
    resized_img_dir = 'ResizedImage/testing_data'
    # list all subdir in the dir
    dogCategories = os.listdir(resized_img_dir)
    # The list of arrays that is being output
    data_list=[]
    # The list of ground truths
    ground_truth_list=[]
    # golden_retriever=0, siberian_husky=1
    
    if shuffle: #shuffle the order
        path_list=[]
        for category in dogCategories:
            # ignore .DS_Store file for mac
            if category=='.DS_Store':
                continue
            # construct paths 
            resized_category_path = os.path.join(resized_img_dir, category)
            
            # list all images
            images = os.listdir(resized_category_path)
            for image_path in images:
                if image_path.endswith('.jpg'):
                    path_list.append(image_path)
        # shuffle the dataset
        random.shuffle(path_list)
        for path in path_list:
            if "husky" in path:# data is husky
                img = plt.imread(os.path.join("ResizedImage/testing_data/siberian_husky/", path)) # constructs the full path to the image
                data_list.append(img[:,:,0]/255)
                ground_truth_list.append(np.array([0,1])) 
            else:
                img = plt.imread(os.path.join("ResizedImage/testing_data/golden_retriever/", path)) # constructs the full path to the image
                data_list.append(img[:,:,0]/255)
                ground_truth_list.append(np.array([1,0])) # data is golden retreiver
            
    else: # store img in order
        ground_truth_idx=np.array([1,0])
        for category in dogCategories:
            # ignore .DS_Store file for mac
            if category=='.DS_Store':
                continue
            # construct paths 
            resized_category_path = os.path.join(resized_img_dir, category)
            
            # list all images
            images = os.listdir(resized_category_path)
            for image_path in images:
                if image_path.endswith('.jpg'):
                    img = plt.imread(os.path.join(resized_category_path, image_path)) # constructs the full path to the image
                    data_list.append(img[:,:,0]/255)
                    ground_truth_list.append(ground_truth_idx)
            # update the idx for next iteration
            ground_truth_idx=np.array([0,1])

    return data_list,ground_truth_list