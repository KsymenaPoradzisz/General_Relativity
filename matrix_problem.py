import Derivatives as Div
import Cheb as Ch
import numpy as np
import math
import argparse # this library is for us to parse arguments via console

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
        return 4*np.exp((x - self.mu)**2 / self.sigma**2) * (x - self.mu) * (2 * (x - self.mu)**2 + 3 * self.sigma**2) / self.sigma**6


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
    sinthsqr = [(1 - u * u) for u in U]

#We also need X matrix, defined as 1/Pi Sin(Pi X)
    
    sinPix = []
    sinPix = [(1/np.pi) * np.sin(np.pi * x) for x in X]
    
#And a "frac" matrix: 
    
    frac = []
    frac = [u/(1-u*u) if (1-u*u) != 0 else 1e+12 for u in U]
#Initial condition for eta -- we need to define it, it needs 2 grids and 3 functions, but gaussians
#are already defined upstairs, so i will use them (this is wrong because not compactified!!!)

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

        return np.reshape(temp,-1)


#initial condition for K^R_theta (NOT COMPACTIFIED but i don't know how to do it if I don't know L!)

    def Kru(X,U):
        NX = len(X)
        NU = len(U)
        temp = np.zeros((NX,NU))
        for i in range(NX):
             for j in range(NU):
                u = U[j]
                x = X[i]
                tempGauss = Gauss(0,0.1) #example of gauss parameters
                df = tempGauss.fprim(x)
                dfminus = tempGauss.fprim(-x)
                ddf = tempGauss.fbis(x)
                ddfzero = tempGauss.fbis(0)
                ddfminus = tempGauss.fbis(-x)
                dddf = temp.Gaussftris(x)
                dddfminus = tempGauss.ftris(-x)
                temp[i,j] = 3./2. * u/(x*x*x) * (-3 * dfminus + 3 * df + x * (-8 * ddf + ddf + ddfminus + x * dddfminus - x * dddf))
        return np.reshape(temp,-1)

#Now all we need is a matrix and RHS vectors and we can the apply a solver, i will assume the convention (X, U), if it does not work one can change i<->j in Kru and eta 

    IdX = np.eye(NR)
    IdU = np.eye(NU)

#upper left matrix -> Du + Du*eta
    
    M11 = np.empty((NR, NU))
    M11 = np.kron(IdX,DU) + np.dot(np.kron(IdX,DU), eta)

#upper right matrix -> Du - Du eta - 2u/(1-u^2)
    
    M12 = np.empty((NR, NU))
    M12 = np.kron(IdX,DU) - np.dot(np.kron(IdX,DU), eta) - 2*np.kron(IdX,frac) 

#lower left matrix -> 1/pi sin(pi x) Dx + 3 - 1/2Pi sin(x pi)

    M21 = np.empty((NR, NU))
    M21 = np.kron(sinPix @ DX, IdU) + 3 * np.kron(IdX,IdU) -1./2. * np.kron(sinPix,IdU)

#lower right matrix -> 1/(2 pi) Sin(pi x) Dx eta

    M22 = np.empty((NR, NU))
    M22 = 1./2.* np.dot(np.kron(sinPix @ DX, IdU), eta)

#Now we should join this together: 

    M = np.empty((2*NR, 2*NU)) 
    M = np.block([[M11,M12],[M21,M22]])

#And now it is time for the RHS 

#Upper part -> (1/pi sin(pi x) Dx + 3).K^r_u

    RHS1 =  np.empty((NR,NU))
    RHS1 = np.dot(np.kron(sinPix @ Dx, IdU) + 3 * np.kron(IdX,IdU) - 1./2. * np.kron( sinPix,IdU), Kru)

#Lower part -> ((1 - u^2) - 2 * u).K^r_u
    
    RHS2 =  np.empty((NR,NU))
    RHS2 = np.dot(np.kron(IdX,sinthr) - 2 * np.kron(IdX, u), Kru)

#Now we could test but i am afraid to type python3 matrixproblem.py because it will probably have a lot of errors xd
    
if __name__== "__main__":
    main()
