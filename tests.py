from Data import Gauss
import numpy as np

#For now this test works only for t = 0, we need to think about how to make it work for t > 0 in the future

class Tests:

    def __init__(self, sigma: float, mu: float = 0, epsilon: float = 1, t: float = 0, L: float = 2/np.pi, mode: str = "eta"):
       self.sigma = sigma
       self.mu = mu
       self.epsilon = epsilon
       self.t = t
       self.L = L

    def Linear(self, x, u):
        match mode: 
            case "eta": 
                #Here I return linear approximation to eta function
                tempGauss = Gauss(self.sigma, self.mu, self.epsilon)
                temp = -1/(2 * self.L**2) * 3 * (u*u-1) * self.epsilon * 1/(np.tan(np.pi*x/2))**2

                return temp * (-4 * tempGauss.fprim(0) + 2 * tempGauss.fprim(-x) + 2 * tempGauss.fprim(x) + tempGauss.tg(x) * tempGauss.fbis(-x) - tempGauss.tg(x) * tempGauss.fbis(x))  
