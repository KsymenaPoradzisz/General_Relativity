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

    def Linear(self, x, u,mode):
        match mode: 
            case "eta": 
                #Here I return linear approximation to eta function
                tempGauss = Gauss(self.sigma, self.mu, self.epsilon)
                temp = -1/(2 * self.L**2) * 3 * (u*u-1) * self.epsilon * 1/(np.tan(np.pi*x/2))**2

                return temp * (-4 * tempGauss.fprim(0) + 2 * tempGauss.fprim(-self.x) + 2 * tempGauss.fprim(self.x) + tempGauss.tg(self.x) * tempGauss.fbis(-self.x) - tempGauss.tg(self.x) * tempGauss.fbis(self.x)) 
            case "Kr_theta/r":
                #Here I return linear approximation to Khat^r_theta/r
                tempGauss = Gauss(self.sigma, self.mu, self.epsilon)
                cot = 1/np.tan(np.pi * self.x/2.)
                return ((3 * cot**3 * ff2_prime) / self.LL**3 +
                        (3 * cot**3 * tempGauss.fprim(self.LL * tempGauss.tg(self.x))) / self.LL**3 -
                        (3 * cot**2 * ff2_bis) / self.LL**2 -
                        (3 * cot**2 * tempGauss.fbis(self.LL * tempGauss.tg(self.x))) / self.LL**2 -
                        (2 * cot * tempGauss.ftris(self.LL * tempGauss.tg(self.x))) / self.LL +
                        (2 * cot * tempGauss.ftris(self.LL * tempGauss.tg(self.x))) / self.LL -
                        tempGauss.ftetra(tempGauss.tg(self.x)) -
                        tempGauss.ftetra(tempGauss.tg(self.x))
                        ) * 2./(3. * u * self.epsilon)
