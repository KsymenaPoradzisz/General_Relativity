import Derivatives as Div
import Cheb as Ch
import numpy as np

class Gauss:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def f(self, x):
        return np.exp((x - self.mu)**2 / self.sigma**2)
    
    def fprim(self, x):
        return np.exp((x - self.mu)**2 / self.sigma**2) * (2 * (x - self.mu)) / self.sigma**2
    
    def fbis(self, x):
        return 2*np.exp((x - self.mu)**2 / self.sigma**2) * ((2 *(x - self.mu)**2 + self.sigma**2 ) / self.sigma**4)
    
    def ftris(self,x):
        return 4*np.exp((x - self.mu)**2 / self.sigma**2) * (x - self.mu) * (2 * (x - self.mu)**2 + 3*self.sigma**2) / self.sigma**6


#Initial condition for eta -- we need to define it, it needs 2 grids and 3 functions, but gaussians
#are already defined upstairs, so i will use them

def eta(X, U):
    NX = len(X)
    NU = len(U)
    temp = np.zeros((NX, NU))
    for i in range(NX):
        for j in range(NU):
            u = U[j]
            x = X[i]
            tempGauss = Gauss(0,0.1) #example of gauss parameters
            f = tempGauss.f(x)
            df = tempGauss.fprim(x)
            dfminus = tempGauss.fprim(-x)
            dfzero = tempGauss.fprim(0.)
            ddf = tempGauss.fbis(x)
            ddfminus = tempGauss.fbis(-x)
            temp[i,j] = -3/2 * (u * u - 1) / (x[i] * x[i]) *(-4. * dfzero + 2. * dfminus + 2. * df + x * ddfminus - x * ddf) 

    return temp


#initial condition for K^R_theta

def Kru(X,U):
    NX = len(X)
    NU = len(U)
    temp = np.zeros((NX,NU))
    for i in range(NX):
        for j in range(NU):
            u = U[j]
            x = X[i]
            tempGauss = Gauss(0,0.1) #example of gauss parameters
            df = temp.Gauss.fprim(x)
            dfminus = temp.Gauss.fprim(-x)
            ddf = temp.Gauss.fbis(x)
            ddfzero = temp.Gauss.fbis(0)
            ddfminus = temp.Gauss.fbis(-x)
            dddf = temp.Gauss.ftris(x)
            dddfminus = temp.Gauss.ftris(-x)
            temp[i,j] = 3./2. * u/(x*x*x) * (-3 * dfminus + 3 * df + x * (-8 * ddf + ddf + ddfminus + x * dddfminus - x * dddf))
    return temp



def main():
    NR = 40
    NU = 20

#Now let's head into the problem, define grids and matrices        
    X = np.empty((NR,))
    U = np.empty((NU,))

    DX = np.empty((NR,NR))
    DU = np.empty((NU, NU))

#Use Derivatives.py and cheb.py to get grid and matrices
#Notice that we should mirror the X grid for better accuracy -- to do later
    
    X = Ch.Grid(NR, mode='cheb').grid
    DX = Div.DR(NR).matrix
    U = Ch.Grid(NU, mode='cheb').grid
    DU = Div.DR(NU).matrix

#costh is just U
#sinth is just sqrt(1-u^2)

    sinth = []
    sinth = [(1 - u * u) ** (1 / 2) for u in U]

#same for sinthsquared

    sinthsqr = []
    sinthsqr = [(1 - u * u) for u in gridT]

#Now all we need is a matrix and RHS vectors and we can then apply a solver

if __name__== "__main__":
    main()
