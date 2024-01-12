import numpy as np
import numpy.typing as npt
import Cheb as Ch
import Derivatives as Div


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
    pass


class Eta(Data):
     # for the sake of numerical stability we choose \Eta/R
     # X - spatial grid; U - angular grid
    def __init__(self, X: list, U: list) -> None:
        self.eta = self.gaussInit(X, U)

    def gaussInit(self, X: list, U: list) -> None:
        NX, NU = len(X), len(U)
        eta = np.zeros((NX, NU))
        gaussian = Gauss(sigma = 0.5, mu = 0, epsilon = 0.00001) #example of gauss parameters
        for i in range(NX):
            for j in range(NU):
                u, r = U[j], X[i]
                f = gaussian.f(r)
                df = gaussian.fprim(r)
                dfminus = gaussian.fprim(-r)
                dfzero = gaussian.fprim(0.)
                ddf = gaussian.fbis(r)
                ddfminus = gaussian.fbis(-r)
                eta[i, j] = -3/2 * (u*u - 1) / (r*r) *(-4.*dfzero + 2.*dfminus + 2.*df + r*ddfminus - r*ddf) if r != 0 else 1e+12 

        return np.reshape(eta, -1)


class Kru(Data):
    # X - spatial grid; U - angular grid
    def __init__(self, X: list, U: list) -> None:
        self.kru = self.gaussInit(X, U)

    def gaussInit(self, X: list, U: list) -> None:
        NX, NU = len(X), len(U)
        kru = np.zeros((NX,NU))
        gaussian = Gauss(sigma = 0.5, mu = 0, epsilon = 0.00001) #example of gauss parameters
        for i in range(NX):
            for j in range(NU):
                u, r = U[j], X[i]
                f = gaussian.fprim(r)
                df = gaussian.fprim(r)
                dfminus = gaussian.fprim(-r)
                ddf = gaussian.fbis(r)
                ddfzero = gaussian.fbis(0)
                ddfminus = gaussian.fbis(-r)
                dddf = gaussian.ftris(r)
                dddfminus = gaussian.ftris(-r)
                kru[i, j] = 3./2. * u/(r*r*r) * (-3 * dfminus + 3 * df + r * (-8 * ddf + ddf + ddfminus + r * dddfminus - r * dddf)) if r != 0 else 1e+12 
        
        return np.reshape(kru, -1)


class KRTheta(Data):
    # for the sake of numerical stability we choose KRTheta/R
    pass


class BetaR(Data):
    # for the sake of numerical stability we choose \BetaR/R
    def dr(self) -> npt.NDArray:
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
