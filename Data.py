import numpy as np
import numpy.typing as npt
import Cheb as Ch
import Derivatives as Div
import math

class Gauss:
    def __init__(self, sigma: float, mu: float = 0, epsilon: float = 1):
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
    def f(self, x):
        return self.epsilon*np.exp((x - self.mu)**2 / self.sigma**2)
    
    def fprim(self, x):
        return self.epsilon*np.exp((x - self.mu)**2 / self.sigma**2) * (2 * (x - self.mu)) / self.sigma**2
    
    def fbis(self, x):
        return 2*self.epsilon*np.exp((x - self.mu)**2 / self.sigma**2) * ((2 *(x - self.mu)**2 + self.sigma**2 ) / self.sigma**4)
    
    def ftris(self, x):
        return 4*self.epsilon*np.exp((x - self.mu)**2 / self.sigma**2) * (x - self.mu) * (2 * (x - self.mu)**2 + 3 * self.sigma**2) / self.sigma**6
class Data: 
    def __init__(self, X: list, U: list) -> None:
        self.NX = len(X)
        self.NU = len(U)
class Eta(Data):
     # for the sake of numerical stability we choose \Eta/R
   # X - x grid; U - u grid
    def __init__(self, X: list, U: list) -> None:
        super(Eta, self).__init__(X, U)
        self.eta = np.zeros((self.NX, self.NU))
    
    def initialize(self, mode = "gauss"):
        match mode:
            case "gauss":
                for i in range(self.NX):
                    for j in range(self.NU):
                        u = U[j]
                        r = X[i]
                        # r = 2./np.pi * np.tan(np.pi * x / 2)
                        tempGauss = Gauss(sigma = 0.5, mu = 0, epsilon = 0.00001) #example of gauss parameters
                        f = tempGauss.f(r)
                        df = tempGauss.fprim(r)
                        dfminus = tempGauss.fprim(-r)
                        dfzero = tempGauss.fprim(0.)
                        ddf = tempGauss.fbis(r)
                        ddfminus = tempGauss.fbis(-r)
                        self.eta[j, i] = -3/2 * (u*u - 1) / (r*r) *(-4.*dfzero + 2.*dfminus + 2.*df + r*ddfminus - r*ddf) if r != 0 else 1e+12 

                self.eta = np.reshape(self.eta, -1)
            case _:
                raise ValueError("There is no mode you chose. Please choose one existing")   

class BetaR(Data):
    # for the sake of numerical stability we choose \BetaR/R
    def dr(self) -> npt.NDArray:
        pass


class KRTheta(Data):
    # for the sake of numerical stability we choose KRTheta/R
    pass


class BetaU(Data):
    # for the sake of numerical stability we choose \BetaU/sqrt(1-u^2)
    pass


class Alpha(Data):
    # for the sake of numerical stability we choose \Alpha/R^2
    pass


class DiagId:
    def __init__(self, diagN: int) -> None:
        self.matrix = np.identity(diagN)

class AntiDiagId:
    def __init__(self, diagN: int) -> None:
        # diagN should be even for matrix to be symmetric!
        self.matrix = np.kron(np.array([[0, 1], [1, 0]]), np.identity(diagN//2))
