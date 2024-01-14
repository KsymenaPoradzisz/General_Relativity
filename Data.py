import numpy as np
import numpy.typing as npt
import Cheb as Ch
import Derivatives as Div


class Gauss:
    def __init__(self, sigma: float, mu: float = 0, epsilon: float = 1, L: float = 2/np.pi):
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
        self.L = L
    #I define tg(x) and cos(x) because they appear in the derivatives

    def tg(self,x):
     #Define compactified variable, L = 2/Pi
        if x > -1 and x < 1:
            tg = np.tan(np.pi * x/2.) * self.L
        elif x == 0:
            tg =  1e+6
        elif abs(x) == 1:
            tg = -1e+6
        else:
            raise ValueError("argument should be in range [0,1]")
        return tg
        

    # Define a derivative of such variable, this is Sec^2 * Pi/2 * L, so just Sec^2 but we can have 
   
   # Different values of L, so i leave that
    def secL(self,x):
        if x >= -1 and x <= 1:
            if(abs(x) != 1):
                return np.pi/2. * 1/(np.cos(np.pi*x/2.))**2 * self.L
            else: 
                return 1e+6
        else:
            raise ValueError("argument should be in range [0,1]")
    #This is the same secans, but without L, because it is convinient for second derivative
    def sec(self,x):
        if x >= -1 and x <= 1:

            if(abs(x) != 1):
                return 1/(np.cos(np.pi*x/2.))**2
            else: 
                return 1e+6
        else:
            raise ValueError("argument should be in range [0,1]")
            


   #Now define the gauss function up to 3rd derivative

    def f(self, x):
        return self.epsilon*np.exp((self.tg(x) - self.mu)**2 / self.sigma**2)
    
    def fprim(self, x):
        return self.f(x) * (-2 * (self.tg(x) - self.mu)) * self.secL(x)

    #Unfortunatelly, other derivatives are really bad :( I write them expanded, so it is easier to check
    
    def fbis(self, x):
        tempe = self.f(x) * self.L * self.L * np.pi**2/self.sigma**2
        return  (-self.sec(x)**2/2. + (self.mu*self.sec(x)/self.sigma)**2 + self.mu * self.tg(x)/self.L * self.sec(x)/self.L - 2 * self.tg(x) * self.mu * self.sec(x)**2/self.sigma**2 - self.sec(x) * (self.tg(x)/self.L)**2 + self.tg(x)**2 * self.sec(x)**2/self.sigma**2) * tempe


    def ftris(self, x):
        tempe = self.f(x) * self.L * np.pi**3/self.sigma**2
        return ( 1./2. * self.mu * self.sec(x)**2 + (
            self.L/self.sigma**2)**2 * (self.mu * self.sec(x))**3 -
            3./2. * (self.L/self.sigma)**2 * self.sec(x)**3 -
            2*self.tg(x) * self.sec(x)**2 + 3 * self.tg(x) * (self.sec(x)/self.sigma)**2 -
            3 * self.tg(x) * self.L**2 * self.mu**2 /self.sigma**4 * self.sec(x)**3 +
            3*self.tg(x) * (self.L/self.sigma)**2 * self.sec(x)**3 /2. +
            self.mu * self.sec(x) * self.tg(x)/self.L**2 -
            6 * (self.tg(x)/self.sigma)**2 * self.sec(x)**2 +
            3 * self.mu * self.tg(x)**2 * self.L**2 /self.sigma**4 * self.sec(x)**3 -
            self.tg(x)**3 * (self.sec(x)/self.L)**2 +3 * self.tg(x)**3 * (self.sec(x)/self.sigma)*2 -
            self.tg(x)**3 * self.L**2/self.sigma**4 * self.sec(x)**3) * tempe 

#IT TURNS OUT YOU ALSO NEED 4TH DERIVATIVE :) 

    def ftetra(self,x):
        tempe = (self.f(x) * self.L**2 * np.pi ** 4)/self.sigma**2 
        return (-self.sec(x)**3 + (2 * self.mu**2 * self.sec(x)**3) / self.sigma**2 + (
            self.L**2 * self.mu**4 * self.sec(x)**4) / self.sigma**6 - (
            3 * self.L**2 * self.mu**2 * self.sec(x)**4) / self.sigma**4 + (
            3 * self.L**2 * self.sec(x)**4) / (4 * self.sigma**2) + (
            2 * self.mu * self.tg(x) * self.sec(x)**4 / self.L**2) + (
            6  * self.mu**3 * self.sec(x)**3 * self.tg(x)) / self.sigma**4 - (
            13 * self.mu * self.sec(x)**3 * self.tg(x)) / self.sigma**2 - (
            4 * self.L**2 * self.mu**3 * self.sec(x)**4 * self.tg(x)) / self.sigma**6 + (
            6 * self.L**2 * self.mu * self.sec(x)**4 * self.tg(x)) / self.sigma**4 - 
            11 / 2 * self.sec(x)**4 * self.tg(x)**2/ + (
            7 * self.mu**2 * self.sec(x)**2 * self.tg(x)**2) / (self.sigma*self.L)**2 - (
            18 * self.mu**2 * self.sec(x)**3 * self.tg(x)**2) / self.sigma**4 + (
            11 * self.sec(x)**3 * self.tg(x)**2) / self.sigma**2 + (
            6 * self.L**2 * self.mu**2 * self.sec(x)**4 * self.tg(x)**2) / self.sigma**6 - (
            3 * self.L**2 * self.sec(x)**4 * self.tg(x)**2) / self.sigma**4 + (
            self.mu * self.sec(x) * self.tg(x)**3) / self.L**4 - (
            14  * self.mu * self.sec(x)**2 * self.tg(x)**3) / (self.L * self.sigma**2) + (
            18  * self.mu * self.sec(x)**3 * self.tg(x)**3) / self.sigma**4 - (
            4 * self.L**2 * self.mu * self.sec(x)**4 * self.tg(x)**3) / self.sigma**6 - 
            self.sec(x) * self.tg(x)**4 + (
            7 / self.L**2 * self.sec(x)**2 * self.tg(x)**4) / self.sigma**2 - (
            6 * self.sec(x)**3 * self.tg(x)**4) / self.sigma**4 + (
            self.L**2 * self.sec(x)**4 * self.tg(x)**4) / self.sigma**6) * tempe



class Data: 
    def __init__(self, X: list, U: list) -> None:
        self.NX = len(X)
        self.NU = len(U)
class Eta(Data):
     # for the sake of numerical stability we choose \Eta/R
   # X - x grid; U - u grid
    def __init__(self, X: list, U: list) -> None:
        super(Eta, self).__init__(X, U)
        self.eta = np.zeros((self.NU, self.NX))
        self.U = U
        self.X = X

    def init_cond(self, mode = "gauss"):
        match mode:
            case "gauss":
                tempGauss = Gauss(sigma = 0.5, mu = 0, epsilon = 0.00001) #example of gauss parameters
                for i in range(self.NX):
                    for j in range(self.NU):
                        u = self.U[j]
                        r = self.X[i]
                        # r = 2./np.pi * np.tan(np.pi * x / 2)
                        f = tempGauss.f(r)
                        df = tempGauss.fprim(r)
                        dfminus = tempGauss.fprim(-r)
                        dfzero = tempGauss.fprim(0.)
                        ddf = tempGauss.fbis(r)
                        ddfminus = tempGauss.fbis(-r)
                        self.eta[j, i] = -3/2 * (u*u - 1) / (r*r) *(-4.*dfzero + 2.*dfminus + 2.*df + r*ddfminus - r*ddf) if r != 0 else 1e+12 

                self.eta = np.reshape(self.eta, -1)
                return self
            case _:
                raise ValueError("There is no mode you chose. Please choose one existing")   

class Kru(Data):
    # X - spatial grid; U - angular grid
    def __init__(self, X: list, U: list) -> None:
        super(Kru, self).__init__(X, U)
        self.kru = np.zeros((self.NX,self.NU))
        self.U = U
        self.X = X


    def init_cond(self, mode = "gauss") -> None:
        match mode:
            case "gauss":
                gaussian = Gauss(sigma = 0.5, mu = 0, epsilon = 0.00001) #example of gauss parameters
                for i in range(self.NX):
                    for j in range(self.NU):
                        u, r = self.U[j], self.X[i]
                        f = gaussian.fprim(r)
                        df = gaussian.fprim(r)
                        dfminus = gaussian.fprim(-r)
                        ddf = gaussian.fbis(r)
                        ddfzero = gaussian.fbis(0)
                        ddfminus = gaussian.fbis(-r)
                        dddf = gaussian.ftris(r)
                        dddfminus = gaussian.ftris(-r)
                        self.kru[i, j] = 3./2. * u/(r*r*r) * (-3 * dfminus + 3 * df + r * (-8 * ddf + ddf + ddfminus + r * dddfminus - r * dddf)) if r != 0 else 1e+12 
                
                self.kru = np.reshape(self.kru, -1)
                return self
            case _:
                raise ValueError("There is no mode you chose. Please choose one existing") 


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
