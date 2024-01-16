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
        tempGauss = Gauss(self.sigma, self.mu, self.epsilon)
        cot = 1/np.tan(np.pi * x/2.)
        csc = 1/np.sin(np.pi * x/2)
        match mode: 
            case "eta": 
                #Here I return linear approximation to eta function
                
                temp = -1/(2 * self.L**2) * 3 * (u*u-1) * self.epsilon * 1/(np.tan(np.pi*x/2))**2

                return temp * (-4 * tempGauss.fprim(0) + 2 * tempGauss.fprim(-x) + 2 * tempGauss.fprim(x) + tempGauss.tg(x) * tempGauss.fbis(-x) - tempGauss.tg(x) * tempGauss.fbis(x)) 
            case "Kr_theta/r":
                #Here I return linear approximation to Khat^r_theta/r
               # tempGauss = Gauss(self.sigma, self.mu, self.epsilon)
                
                return ((3 * cot**3 * tempGauss.fprim) / self.L**3 +
                        (3 * cot**3 * tempGauss.fprim(self.L * tempGauss.tg(x))) / self.L**3 -
                        (3 * cot**2 * tempGauss.fbis) / self.L**2 -
                        (3 * cot**2 * tempGauss.fbis(self.L * tempGauss.tg(x))) / self.L**2 -
                        (2 * cot * tempGauss.ftris(self.L * tempGauss.tg(x))) / self.L +
                        (2 * cot * tempGauss.ftris(self.L * tempGauss.tg(x))) / self.L -
                        tempGauss.ftetra(tempGauss.tg(x)) -
                        tempGauss.ftetra(tempGauss.tg(x))
                        ) * 2./(3. * u * self.epsilon)
            case "Krr":
                #ERR = FF[2] u rostwora
                expression = (1 / (2 * self.L**5)) * 3 * (-1 + 3 * u**2) * self.epsilon * cot**3 * csc**2 * (
                -3 * self.L * np.sin(np.pi * x) * tempGauss.f(self.t - self.L * np.tan((np.pi * x) / 2)) -
                3 * self.L * np.sin(np.pi * x) * tempGauss.f(self.t + self.L * np.tan((np.pi * x) / 2)) -
                3 * ERR(self.t - self.L * np.tan((np.pi * x) / 2)) -
                3 * np.cos(np.pi * x) * ERR(self.t - self.L * np.tan((np.pi * x) / 2)) +
                3 * ERR(self.t + self.L * np.tan((np.pi * x) / 2)) +
                3 * np.cos(np.pi * x) * FF[2](t + self.L * np.tan((np.pi * x) / 2)) -
                self.L**2 * tempGauss.fprim(self.t - self.L * np.tan((np.pi * x) / 2)) +
                self.L**2 * np.cos(np.pi * x) * tempGauss.fprim(self.t - self.L * np.tan((np.pi * x) / 2)) +
                self.L**2 * tempGauss.fprim(self.t + self.L * np.tan((np.pi * x) / 2)) -
                self.L**2 * np.cos(np.pi * x) * tempGauss.fprim(self.t + self.L * np.tan((np.pi * x) / 2)))
                return expression




