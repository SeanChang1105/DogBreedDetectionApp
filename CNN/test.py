import initialModel
import numpy as np
import matplotlib.pyplot as plt
import dataLoader
from crossEntropy import crossEntropy
import testingModels
import sys


trainX,trainY=dataLoader.loadTrainingData(True)
testX,testY=dataLoader.loadTestingData(True)

epoch=10
model=testingModels.model3()
lossFunc=crossEntropy.cross_entropy_loss
learning_rate=0.001
lossList=[]


for e in range(epoch):
    epochLoss=0
    print("epoch: ",e+1)
    idx=0
    for x,y in zip(trainX,trainY):
        idx+=1
        prediction=model.forward(x)
        loss=lossFunc(y,prediction)
        epochLoss+=loss
        model.backward(loss,y,learning_rate)
        if idx%10==0:
            print(loss)
    lossList.append(epochLoss/len(trainX))
    print("Loss of this epoch: ",epochLoss)

correct=0
for x,y in zip(testX,testY):
    prediction=model.forward(x)
    if prediction[0]>prediction[1]:
        prediction=np.array([1,0])
    else:
        prediction=np.array([0,1])
    print("p",prediction)
    print("y",y)
    if np.array_equal(y,prediction):
        correct+=1
acc=correct/len(testX)
print("accuracy: ",acc)

xaxis=range(1,epoch+1)
plt.plot(xaxis,lossList)
plt.xlabel("Epochs")
plt.ylabel("Avg Loss")
plt.show()

# pred=model.forward(testX[0])
# print(pred)
# print(testY[0])
# plt.imshow(testX[0].get(),cmap='grey')
# plt.show()

        

