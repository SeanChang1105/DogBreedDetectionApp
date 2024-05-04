import numpy as np
class crossEntropy:
    def cross_entropy_loss(yTrue,yPrediction):
        e=1e-10
        yPrediction=np.clip(yPrediction,e,1-e) # clip the value for numerical stability
        loss=-np.sum(yTrue*np.log(yPrediction))
        return loss