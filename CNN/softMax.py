import numpy as np
class SoftMax:
    def __init__(self):
        self.yPredict=None
        self.yTrue=None

    def forward(self,input):
        e_x=np.exp(input-np.max(input)) #subtract the maximum for numerical stabilization
        self.yPredict= e_x/np.sum(e_x)
        return self.yPredict

    def backward(self,yTrue):
        self.yTrue=yTrue
        dLdx=np.array([x-y for x,y in zip(self.yPredict,self.yTrue)])
        return dLdx