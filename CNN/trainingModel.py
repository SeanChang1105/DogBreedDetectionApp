from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import math
import convolution
import relu
import maxPool
import fullyConnectedLayer
import softMax
import dataLoader
from crossEntropy import crossEntropy
from initialModel import model1
import os

class trainingModel:
    def __init__(self, epochs) -> None:
        self.epochs = epochs
        self.average_loss_after_each_epoch_test = [] # initialize array that tracks the average loss after each epoch when the testing set is different from the training set
        self.average_accuracy_after_each_epoch_test = [] # initialize array that tracks the average accuracy after each epoch when the testing set is different from the training set
        self.model = model1()
        
    def trainModel(self, images_train, training_labels, images_test, testing_labels, learning_rate):
        for i in range(self.epochs):
            print(f"Epoch: {i+1}/{self.epochs}") # print epoch number for debugging purposes and to see if the code is running 
            loss_test = self.train(images_train, training_labels, learning_rate) # train the model on the training set of images 
            print(f"Loss: {loss_test}")
            self.average_loss_after_each_epoch_test.append(loss_test)

            accuracy_test = self.test(images_test, testing_labels)
            print(f"Accuracy: {accuracy_test}")
            self.average_accuracy_after_each_epoch_test.append(accuracy_test)
            
    def train(self, images_train, training_labels, learning_rate):
        total_loss = 0
        for i in range(len(images_train)):
            model_prediction = self.model.forward(images_train[i])
            if training_labels[i] == 0:
                training_label = np.array([1,0])
            else:
                training_label = np.array([0,1])
            loss=crossEntropy.cross_entropy_loss(training_label,model_prediction)
            total_loss += loss
            self.model.backward(loss, training_label, learning_rate)
        total_loss /= len(images_train)
        return total_loss
            
    def test(self, images_test, testing_labels):
        accuracy = 0
        for i in range(len(images_test)):
            model_prediction = self.model.forward(images_test[i])
            max_index = np.argmax(model_prediction)
            abs_model_prediction = np.zeros_like(model_prediction)
            abs_model_prediction[max_index] = 1

            if testing_labels[i] == 0:
                testing_label = np.array([1,0])
            else:
                testing_label = np.array([0,1])

            if np.array_equal(abs_model_prediction, testing_label):
                accuracy += 1

        accuracy /= len(images_test)
        return accuracy
            
    def plot(self):
        # Plot accuracy
        plt.plot(list(range(1, len(self.average_accuracy_after_each_epoch_test) + 1)), self.average_accuracy_after_each_epoch_test) # Plot average accuracy for each epoch for when the testing set is different from the training set
        plt.xlabel('Epoch') # Change the X-axis label to 'Epoch'
        plt.ylabel('Accuracy') # Change the Y-axis label to 'Accuracy'
        plt.show() # display the graph in the output

        # Plot loss
        plt.plot(list(range(1, len(self.average_loss_after_each_epoch_test) + 1)), self.average_loss_after_each_epoch_test) #  Plot average loss over for epoch for when the testing set is different from the training set
        plt.xlabel('Epoch') # Change the X-axis label to 'Epoch'
        plt.ylabel('Loss') # Change the Y-axis label to 'Loss'
        plt.show() # display the graph in the output

    def predict(self, image):
        img = plt.imread(image)
        img = img[:,:,0]
        output = self.model.forward(img)
        output_class = np.argmax(output)
        dog_ident = ['Golden Retriever', 'Husky']
        print(f"predicted: {dog_ident[output_class]}")

    def saveModel(self, file='modelParameters.txt'):
        self.model.saveModel(file)

    def loadModel(self, file='modelParameters.txt'):
        self.model.loadModel(file)

if __name__ == '__main__':
    first_model = trainingModel(epochs=5)

    # Want to train model
    images_train, training_labels = dataLoader.loadTrainingData(shuffle=True)
    images_test, testing_labels = dataLoader.loadTestingData(shuffle=True)

    learning_rate = 0.01

    first_model.trainModel(images_train, training_labels, images_test, testing_labels, learning_rate)
    first_model.plot()

    print("--Golden Retriver--")
    first_model.predict('ResizedImage/testing_data/golden_retriever/resize_gr3.jpg')
    print("--Husky--")
    first_model.predict('ResizedImage/testing_data/siberian_husky/resize_husky7.jpg')
    
    first_model.saveModel()

    # Want to load model and test it
    # first_model.loadModel('test.txt')
    # print("--Golden Retriver--")
    # first_model.predict('ResizedImage/testing_data\golden_retriever/resize_gr74.jpg')
    # print("--Husky--")
    # first_model.predict('ResizedImage/testing_data\siberian_husky/resize_husky86.jpg')


