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
import layerNorm
from crossEntropy import crossEntropy
from initialModel import model1
import os

class test2:
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
        #self.fullyConnectedLayer2=fullyConnectedLayer.fullyConnectedLayer(4096,2)
        self.softMaxLayer=softMax.SoftMax()
        self.layerNorm = layerNorm.layerNorm()
        
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
        # convolution
        cov_out1=self.convolutionLayer1.forward(img)
        cov_out2=self.convolutionLayer2.forward(img)
        cov_out3=self.convolutionLayer3.forward(img)
        cov_out4=self.convolutionLayer4.forward(img)
        
        # layer norm
        relu_out1_pre = self.layerNorm.forward(cov_out1)
        relu_out2_pre = self.layerNorm.forward(cov_out2)
        relu_out3_pre = self.layerNorm.forward(cov_out3)
        relu_out4_pre = self.layerNorm.forward(cov_out4)
        # relu
        relu_out1=self.reluLayer.forward(relu_out1_pre)
        relu_out2=self.reluLayer.forward(relu_out2_pre)
        relu_out3=self.reluLayer.forward(relu_out3_pre)
        relu_out4=self.reluLayer.forward(relu_out4_pre)
        
        # maxpool
        pool_out1=self.maxPoolLayer.forward(relu_out1)
        pool_out2=self.maxPoolLayer.forward(relu_out2)
        pool_out3=self.maxPoolLayer.forward(relu_out3)
        pool_out4=self.maxPoolLayer.forward(relu_out4)
        
        # another conv
        cov_out1_1=self.convolutionLayer1.forward(pool_out1)
        cov_out2_1=self.convolutionLayer2.forward(pool_out1)
        cov_out3_1=self.convolutionLayer3.forward(pool_out1)
        cov_out4_1=self.convolutionLayer4.forward(pool_out1)
        
        cov_out1_2=self.convolutionLayer1.forward(pool_out2)
        cov_out2_2=self.convolutionLayer2.forward(pool_out2)
        cov_out3_2=self.convolutionLayer3.forward(pool_out2)
        cov_out4_2=self.convolutionLayer4.forward(pool_out2)
        
        cov_out1_3=self.convolutionLayer1.forward(pool_out3)
        cov_out2_3=self.convolutionLayer2.forward(pool_out3)
        cov_out3_3=self.convolutionLayer3.forward(pool_out3)
        cov_out4_3=self.convolutionLayer4.forward(pool_out3)
        
        cov_out1_4=self.convolutionLayer1.forward(pool_out4)
        cov_out2_4=self.convolutionLayer2.forward(pool_out4)
        cov_out3_4=self.convolutionLayer3.forward(pool_out4)
        cov_out4_4=self.convolutionLayer4.forward(pool_out4)
        
        # layer norm
        relu2_out1_1_pre = self.layerNorm.forward(cov_out1_1)
        relu2_out2_1_pre = self.layerNorm.forward(cov_out2_1)
        relu2_out3_1_pre = self.layerNorm.forward(cov_out3_1)
        relu2_out4_1_pre = self.layerNorm.forward(cov_out4_1)
        
        relu2_out1_2_pre = self.layerNorm.forward(cov_out1_2)
        relu2_out2_2_pre = self.layerNorm.forward(cov_out2_2)
        relu2_out3_2_pre = self.layerNorm.forward(cov_out3_2)
        relu2_out4_2_pre = self.layerNorm.forward(cov_out4_2)
        
        relu2_out1_3_pre = self.layerNorm.forward(cov_out1_3)
        relu2_out2_3_pre = self.layerNorm.forward(cov_out2_3)
        relu2_out3_3_pre = self.layerNorm.forward(cov_out3_3)
        relu2_out4_3_pre = self.layerNorm.forward(cov_out4_3)
        
        relu2_out1_4_pre = self.layerNorm.forward(cov_out1_4)
        relu2_out2_4_pre = self.layerNorm.forward(cov_out2_4)
        relu2_out3_4_pre = self.layerNorm.forward(cov_out3_4)
        relu2_out4_4_pre = self.layerNorm.forward(cov_out4_4)
        
        # relu
        relu2_out1_1=self.reluLayer.forward(relu2_out1_1_pre)
        relu2_out1_2=self.reluLayer.forward(relu2_out1_2_pre)
        relu2_out1_3=self.reluLayer.forward(relu2_out1_3_pre)
        relu2_out1_4=self.reluLayer.forward(relu2_out1_4_pre)
        
        relu2_out2_1=self.reluLayer.forward(relu2_out2_1_pre)
        relu2_out2_2=self.reluLayer.forward(relu2_out2_2_pre)
        relu2_out2_3=self.reluLayer.forward(relu2_out2_3_pre)
        relu2_out2_4=self.reluLayer.forward( relu2_out2_4_pre)
        
        relu2_out3_1=self.reluLayer.forward(relu2_out3_1_pre)
        relu2_out3_2=self.reluLayer.forward(relu2_out3_2_pre)
        relu2_out3_3=self.reluLayer.forward(relu2_out3_3_pre)
        relu2_out3_4=self.reluLayer.forward(relu2_out3_4_pre)
        
        relu2_out4_1=self.reluLayer.forward(relu2_out4_1_pre)
        relu2_out4_2=self.reluLayer.forward(relu2_out4_2_pre)
        relu2_out4_3=self.reluLayer.forward(relu2_out4_3_pre)
        relu2_out4_4=self.reluLayer.forward(relu2_out4_4_pre)
        
        
        # maxpool
        pool2_out1_1 = self.maxPoolLayer.forward(relu2_out1_1)
        pool2_out1_2 = self.maxPoolLayer.forward(relu2_out1_2)
        pool2_out1_3 = self.maxPoolLayer.forward(relu2_out1_3)
        pool2_out1_4 = self.maxPoolLayer.forward(relu2_out1_4)
        
        pool2_out2_1 = self.maxPoolLayer.forward(relu2_out2_1)
        pool2_out2_2 = self.maxPoolLayer.forward(relu2_out2_2)
        pool2_out2_3 = self.maxPoolLayer.forward(relu2_out2_3)
        pool2_out2_4 = self.maxPoolLayer.forward(relu2_out2_4) 
        
        pool2_out3_1 = self.maxPoolLayer.forward(relu2_out3_1)
        pool2_out3_2 = self.maxPoolLayer.forward(relu2_out3_2)
        pool2_out3_3 = self.maxPoolLayer.forward(relu2_out3_3)
        pool2_out3_4 = self.maxPoolLayer.forward(relu2_out3_4) 
        
        pool2_out4_1 = self.maxPoolLayer.forward(relu2_out4_1)
        pool2_out4_2 = self.maxPoolLayer.forward(relu2_out4_2)
        pool2_out4_3 = self.maxPoolLayer.forward(relu2_out4_3)
        pool2_out4_4 = self.maxPoolLayer.forward(relu2_out4_4) 
        
        # fcl
        full_input1_1=pool2_out1_1.flatten()
        full_input1_2=pool2_out1_2.flatten()
        full_input1_3=pool2_out1_3.flatten()
        full_input1_4=pool2_out1_4.flatten()
        full_input2_1=pool2_out2_1.flatten()
        full_input2_2=pool2_out2_2.flatten()
        full_input2_3=pool2_out2_3.flatten()
        full_input2_4=pool2_out2_4.flatten()
        full_input3_1=pool2_out3_1.flatten()
        full_input3_2=pool2_out3_2.flatten()
        full_input3_3=pool2_out3_3.flatten()
        full_input3_4=pool2_out3_4.flatten()
        full_input4_1=pool2_out4_1.flatten()
        full_input4_2=pool2_out4_2.flatten()
        full_input4_3=pool2_out4_3.flatten()
        full_input4_4=pool2_out4_4.flatten()
        full = np.concatenate((full_input1_1, full_input2_1, full_input3_1, full_input4_1, full_input1_2, full_input2_2, full_input3_2, full_input4_2, full_input1_3, full_input2_3, full_input3_3, full_input4_3, full_input1_4, full_input2_4, full_input3_4, full_input4_4))
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
        pool2_input1_1 = full_gradient[0:256].reshape(16,16)
        pool2_input2_1 = full_gradient[256:512].reshape(16,16)
        pool2_input3_1 = full_gradient[512:768].reshape(16,16)
        pool2_input4_1 = full_gradient[768:1024].reshape(16,16)
        pool2_input1_2 = full_gradient[1024:1280].reshape(16,16)
        pool2_input2_2 = full_gradient[1280:1536].reshape(16,16)
        pool2_input3_2 = full_gradient[1536:1792].reshape(16,16)
        pool2_input4_2 = full_gradient[1792:2048].reshape(16,16)
        pool2_input1_3 = full_gradient[2048:2304].reshape(16,16)
        pool2_input2_3 = full_gradient[2304:2560].reshape(16,16)
        pool2_input3_3 = full_gradient[2560:2816].reshape(16,16)
        pool2_input4_3 = full_gradient[2816:3072].reshape(16,16)
        pool2_input1_4 = full_gradient[3072:3328].reshape(16,16)
        pool2_input2_4 = full_gradient[3328:3584].reshape(16,16)
        pool2_input3_4 = full_gradient[3584:3840].reshape(16,16)
        pool2_input4_4 = full_gradient[3840:4096].reshape(16,16)
        
        pool2_gradient1_1=self.maxPoolLayer.backward(pool2_input1_1)
        pool2_gradient2_1=self.maxPoolLayer.backward(pool2_input2_1)
        pool2_gradient3_1=self.maxPoolLayer.backward(pool2_input3_1)
        pool2_gradient4_1=self.maxPoolLayer.backward(pool2_input4_1)
        pool2_gradient1_2=self.maxPoolLayer.backward( pool2_input1_2)
        pool2_gradient2_2=self.maxPoolLayer.backward( pool2_input2_2)
        pool2_gradient3_2=self.maxPoolLayer.backward( pool2_input3_2)
        pool2_gradient4_2=self.maxPoolLayer.backward( pool2_input4_2)
        pool2_gradient1_3=self.maxPoolLayer.backward( pool2_input1_3)
        pool2_gradient2_3=self.maxPoolLayer.backward(pool2_input2_3)
        pool2_gradient3_3=self.maxPoolLayer.backward(pool2_input3_3)
        pool2_gradient4_3=self.maxPoolLayer.backward(pool2_input4_3)
        pool2_gradient1_4=self.maxPoolLayer.backward(pool2_input1_4)
        pool2_gradient2_4=self.maxPoolLayer.backward(pool2_input2_4)
        pool2_gradient3_4=self.maxPoolLayer.backward(pool2_input3_4)
        pool2_gradient4_4=self.maxPoolLayer.backward(pool2_input4_4)
        
        relu2_pre1_1 = self.layerNorm.backward( pool2_gradient1_1)
        relu2_pre2_1 = self.layerNorm.backward( pool2_gradient2_1)
        relu2_pre3_1 = self.layerNorm.backward( pool2_gradient3_1)
        relu2_pre4_1 = self.layerNorm.backward( pool2_gradient4_1)
        relu2_pre1_2 = self.layerNorm.backward( pool2_gradient1_2)
        relu2_pre2_2 = self.layerNorm.backward( pool2_gradient2_2)
        relu2_pre3_2 = self.layerNorm.backward( pool2_gradient3_2)
        relu2_pre4_2 = self.layerNorm.backward( pool2_gradient4_2)
        relu2_pre1_3 = self.layerNorm.backward( pool2_gradient1_3)
        relu2_pre2_3 = self.layerNorm.backward( pool2_gradient2_3)
        relu2_pre3_3 = self.layerNorm.backward( pool2_gradient3_3)
        relu2_pre4_3 = self.layerNorm.backward( pool2_gradient4_3)
        relu2_pre1_4 = self.layerNorm.backward( pool2_gradient1_4)
        relu2_pre2_4 = self.layerNorm.backward( pool2_gradient2_4)
        relu2_pre3_4 = self.layerNorm.backward( pool2_gradient3_4)
        relu2_pre4_4 = self.layerNorm.backward( pool2_gradient4_4)
        # relu
        relu2_gradient1_1=self.reluLayer.backward( relu2_pre1_1)
        relu2_gradient2_1=self.reluLayer.backward( relu2_pre2_1)
        relu2_gradient3_1=self.reluLayer.backward( relu2_pre3_1)
        relu2_gradient4_1=self.reluLayer.backward( relu2_pre4_1)
        relu2_gradient1_2=self.reluLayer.backward( relu2_pre1_2)
        relu2_gradient2_2=self.reluLayer.backward( relu2_pre2_2)
        relu2_gradient3_2=self.reluLayer.backward( relu2_pre3_2)
        relu2_gradient4_2=self.reluLayer.backward( relu2_pre4_2)
        relu2_gradient1_3=self.reluLayer.backward( relu2_pre1_3)
        relu2_gradient2_3=self.reluLayer.backward( relu2_pre2_3)
        relu2_gradient3_3=self.reluLayer.backward(relu2_pre3_3)
        relu2_gradient4_3=self.reluLayer.backward(relu2_pre4_3)
        relu2_gradient1_4=self.reluLayer.backward( relu2_pre1_4)
        relu2_gradient2_4=self.reluLayer.backward( relu2_pre2_4)
        relu2_gradient3_4=self.reluLayer.backward( relu2_pre3_4)
        relu2_gradient4_4=self.reluLayer.backward(relu2_pre4_4)
        
        # conv
        cov2_gradient1=self.convolutionLayer1.backward(relu2_gradient1_1,learning_rate)
        cov2_gradient2=self.convolutionLayer2.backward(relu2_gradient2_1,learning_rate)
        cov2_gradient3=self.convolutionLayer3.backward(relu2_gradient3_1,learning_rate)
        cov2_gradient4=self.convolutionLayer4.backward(relu2_gradient4_1,learning_rate)
       
        cov2_gradient5=self.convolutionLayer1.backward(relu2_gradient1_2,learning_rate)
        cov2_gradient6=self.convolutionLayer2.backward(relu2_gradient2_2,learning_rate)
        cov2_gradient7=self.convolutionLayer3.backward(relu2_gradient3_2,learning_rate)
        cov2_gradient8=self.convolutionLayer4.backward(relu2_gradient4_2,learning_rate)
        
        cov2_gradient9=self.convolutionLayer1.backward(relu2_gradient1_3,learning_rate)
        cov2_gradient10=self.convolutionLayer2.backward(relu2_gradient2_3,learning_rate)
        cov2_gradient11=self.convolutionLayer3.backward(relu2_gradient3_3,learning_rate)
        cov2_gradient12=self.convolutionLayer4.backward(relu2_gradient4_3,learning_rate)
        
        cov2_gradient13=self.convolutionLayer1.backward(relu2_gradient1_4,learning_rate)
        cov2_gradient14=self.convolutionLayer2.backward(relu2_gradient2_4,learning_rate)
        cov2_gradient15=self.convolutionLayer3.backward(relu2_gradient3_4,learning_rate)
        cov2_gradient16=self.convolutionLayer4.backward(relu2_gradient4_4,learning_rate)
       
        
         # maxpool
        pool_grdient1_input1 = np.add(cov2_gradient1, cov2_gradient2)
        pool_grdient1_input2 = np.add(cov2_gradient3, cov2_gradient4)
        pool_gradient1_input = np.add(pool_grdient1_input1, pool_grdient1_input2)
        
        pool_grdient2_input1 = np.add(cov2_gradient5, cov2_gradient6)
        pool_grdient2_input2 = np.add(cov2_gradient7, cov2_gradient8)
        pool_gradient2_input = np.add(pool_grdient2_input1, pool_grdient2_input2)
        
        pool_grdient3_input1 = np.add(cov2_gradient9, cov2_gradient10)
        pool_grdient3_input2 = np.add(cov2_gradient11, cov2_gradient12)
        pool_gradient3_input = np.add(pool_grdient3_input1, pool_grdient3_input2)
        
        pool_grdient4_input1 = np.add(cov2_gradient13, cov2_gradient14)
        pool_grdient4_input2 = np.add(cov2_gradient15, cov2_gradient16)
        pool_gradient4_input = np.add(pool_grdient4_input1, pool_grdient4_input2)
        
        pool_gradient1=self.maxPoolLayer.backward(pool_gradient1_input)
        pool_gradient2=self.maxPoolLayer.backward(pool_gradient2_input)
        pool_gradient3=self.maxPoolLayer.backward( pool_gradient3_input)
        pool_gradient4=self.maxPoolLayer.backward( pool_gradient4_input)
        
        # layer norm
        relu_pre1 = self.layerNorm.backward(pool_gradient1)
        relu_pre2 = self.layerNorm.backward(pool_gradient2)
        relu_pre3 = self.layerNorm.backward(pool_gradient3)
        relu_pre4 = self.layerNorm.backward(pool_gradient4)
        
        # relu
        relu_gradient1=self.reluLayer.backward(relu_pre1)
        relu_gradient2=self.reluLayer.backward(relu_pre2)
        relu_gradient3=self.reluLayer.backward(relu_pre3)
        relu_gradient4=self.reluLayer.backward(relu_pre4)
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
            
    def plot(self, filename="Accuracy_test2.png", filename2="Loss_test2.png"):
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
    first_model = test2(epochs=10)

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

   


