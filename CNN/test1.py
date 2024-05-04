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

class test1:
    def __init__(self, epochs) -> None:
        self.epochs = epochs
        self.average_loss_after_each_epoch_test = [] # initialize array that tracks the average loss after each epoch when the testing set is different from the training set
        self.average_accuracy_after_each_epoch_test = [] # initialize array that tracks the average accuracy after each epoch when the testing set is different from the training set
        #self.model = model1()
        self.convolutionLayer1=convolution.ConvolutionLayer()
        self.convolutionLayer2=convolution.ConvolutionLayer()
        self.convolutionLayer3=convolution.ConvolutionLayer()
        self.convolutionLayer4=convolution.ConvolutionLayer()
        self.reluLayer=relu.Relu()
        self.maxPoolLayer=maxPool.MaxPool2D()
        self.fullyConnectedLayer=fullyConnectedLayer.fullyConnectedLayer(4096,2)
        self.softMaxLayer=softMax.SoftMax()
        
    def trainModel(self, images_train, training_labels, images_test, testing_labels, learning_rate):
        for i in range(self.epochs):
            print(f"Epoch: {i+1}/{self.epochs}") # print epoch number for debugging purposes and to see if the code is running 
            loss_test = self.train(images_train, training_labels, learning_rate) # train the model on the training set of images 
            print(f"Loss: {loss_test}")
            self.average_loss_after_each_epoch_test.append(loss_test)

            accuracy_test = self.test(images_test, testing_labels)
            print(f"Accuracy: {accuracy_test}")
            self.average_accuracy_after_each_epoch_test.append(accuracy_test)
            
    def forward(self, img):
        #img = img.astype(float) / 255.0
        # convolution
        cov_out1=self.convolutionLayer1.forward(img)
        cov_out2=self.convolutionLayer2.forward(img)
        cov_out3=self.convolutionLayer3.forward(img)
        cov_out4=self.convolutionLayer4.forward(img)
        
        # relu
        relu_out1=self.reluLayer.forward(cov_out1)
        relu_out2=self.reluLayer.forward(cov_out2)
        relu_out3=self.reluLayer.forward(cov_out3)
        relu_out4=self.reluLayer.forward(cov_out4)
        
        # maxpool
        pool_out1=self.maxPoolLayer.forward(relu_out1)
        pool_out2=self.maxPoolLayer.forward(relu_out2)
        pool_out3=self.maxPoolLayer.forward(relu_out3)
        pool_out4=self.maxPoolLayer.forward(relu_out4)
        
        # fcl
        full_input1=pool_out1.flatten()
        full_input2=pool_out2.flatten()
        full_input3=pool_out3.flatten()
        full_input4=pool_out4.flatten()
        full = np.concatenate((full_input1, full_input2, full_input3, full_input4))
        full_out=self.fullyConnectedLayer.forward(full)
        
        #softmax
        soft_out=self.softMaxLayer.forward(full_out)
        return soft_out
    
    def backward(self, y_true, learning_rate):
        #softmax
        soft_gradient=self.softMaxLayer.backward(y_true)
        # fcl
        full_gradient=self.fullyConnectedLayer.backward(learning_rate,soft_gradient)
        # maxpool
        pool_input1 = full_gradient[0:1024].reshape(32,32)
        pool_input2 = full_gradient[1024:2048].reshape(32,32)
        pool_input3 = full_gradient[2048:3072].reshape(32,32)
        pool_input4 = full_gradient[3072:4096].reshape(32,32)
        pool_gradient1=self.maxPoolLayer.backward(pool_input1)
        pool_gradient2=self.maxPoolLayer.backward(pool_input2)
        pool_gradient3=self.maxPoolLayer.backward(pool_input3)
        pool_gradient4=self.maxPoolLayer.backward(pool_input4)
        # relu
        relu_gradient1=self.reluLayer.backward(pool_gradient1)
        relu_gradient2=self.reluLayer.backward(pool_gradient2)
        relu_gradient3=self.reluLayer.backward(pool_gradient3)
        relu_gradient4=self.reluLayer.backward(pool_gradient4)
        # conv
        cov_gradient1=self.convolutionLayer1.backward(relu_gradient1,learning_rate)
        cov_gradient2=self.convolutionLayer2.backward(relu_gradient2,learning_rate)
        cov_gradient3=self.convolutionLayer3.backward(relu_gradient3,learning_rate)
        cov_gradient4=self.convolutionLayer4.backward(relu_gradient4,learning_rate)
        
            
    def train(self, images_train, training_labels, learning_rate):
        total_loss = 0
        for i in range(len(images_train)):
            model_prediction = self.forward(images_train[i])
            if training_labels[i] == 0:
                training_label = np.array([1,0])
            else:
                training_label = np.array([0,1])
            loss=crossEntropy.cross_entropy_loss(training_label,model_prediction)
            total_loss += loss
            self.backward(training_label, learning_rate)
        total_loss /= len(images_train)
        return total_loss

            
    def test(self, images_test, testing_labels):
        accuracy = 0
        for i in range(len(images_test)):
            model_prediction = self.forward(images_test[i])
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
            
    def plot(self, filename="Accuracy_test1.png", filename2="Loss_test1.png"):
        # Plot accuracy
        plt.plot(list(range(1, len(self.average_accuracy_after_each_epoch_test) + 1)), self.average_accuracy_after_each_epoch_test) # Plot average accuracy for each epoch for when the testing set is different from the training set
        plt.xlabel('Epoch') # Change the X-axis label to 'Epoch'
        plt.ylabel('Accuracy') # Change the Y-axis label to 'Accuracy'
        plt.savefig(filename)
        plt.show() # display the graph in the output

        # Plot loss
        plt.plot(list(range(1, len(self.average_loss_after_each_epoch_test) + 1)), self.average_loss_after_each_epoch_test) #  Plot average loss over for epoch for when the testing set is different from the training set
        plt.xlabel('Epoch') # Change the X-axis label to 'Epoch'
        plt.ylabel('Loss') # Change the Y-axis label to 'Loss'
        plt.savefig(filename2)
        plt.show() # display the graph in the output

    def predict(self, image):
        img = plt.imread(image)
        img = img[:,:,0]
        output = self.forward(img)
        output_class = np.argmax(output)
        dog_ident = ['Golden Retriever', 'Husky']
        print(f"predicted: {dog_ident[output_class]}")

    #def saveModel(self, file='modelParameters.txt'):
    #    self.saveModel(file)

    #def loadModel(self, file='modelParameters.txt'):
    #    self.loadModel(file)

if __name__ == '__main__':
    first_model = test1(epochs=20)

    # Want to train model
    images_train, training_labels = dataLoader.loadTrainingData(shuffle=True)
    images_test, testing_labels = dataLoader.loadTestingData(shuffle=True)

    learning_rate = 0.01

    first_model.trainModel(images_train, training_labels, images_test, testing_labels, learning_rate)
    first_model.plot()

    print("--Golden Retriver--")
    first_model.predict('ResizedImage/testing_data/golden_retriever/resize_gr3.jpg')
    print("--Golden Retriver--")
    first_model.predict('ResizedImage/testing_data/golden_retriever/resize_gr5.jpg')
    print("--Golden Retriver--")
    first_model.predict('ResizedImage/testing_data/golden_retriever/resize_gr9.jpg')
    print("--Husky--")
    first_model.predict('ResizedImage/testing_data/siberian_husky/resize_husky7.jpg')
    print("--Husky--")
    first_model.predict('ResizedImage/testing_data/siberian_husky/resize_husky1.jpg')
    print("--Husky--")
    first_model.predict('ResizedImage/testing_data/siberian_husky/resize_husky4.jpg')
    
    #first_model.saveModel()

   


