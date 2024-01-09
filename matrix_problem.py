import Derivatives as Div
import Cheb as Ch
import numpy as np
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

#Initial condition for eta -- we need to define it, it needs 2 grids and 3 functions, but gaussians
#are already defined above, so i will use them (this is wrong because not compactified!!!)

def Eta(X: list, U: list):
    NX = len(X)
    NU = len(U)
    eta = np.zeros((NU, NX))
    for i in range(NX):
        for j in range(NU):
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
            # print("u = ", u)
            # print("r = ", r)
            # print("dfzero = ", dfzero)
            # print("dfminus = ", dfminus)
            # print("df = ", df)
            # print("f = ", f)
            # print("ddf = ", ddf)
            # print("ddfminus = ", ddfminus)
            eta[j, i] = -3/2 * (u*u - 1) / (r*r) *(-4.*dfzero + 2.*dfminus + 2.*df + r*ddfminus - r*ddf) if r != 0 else 1e+12 

    return np.reshape(eta, -1)


def Kru(X: list, U: list):
    NX = len(X)
    NU = len(U)
    temp = np.zeros((NU,NX))
    for i in range(NX):
        for j in range(NU):
            u = U[j]
            r = X[i]
            #r = 2./np.pi * np.tan(np.pi * x / 2)
            tempGauss = Gauss(sigma = 0.5, mu = 0, epsilon = 0.00001) #example of gauss parameters
            f = tempGauss.fprim(r)
            df = tempGauss.fprim(r)
            dfminus = tempGauss.fprim(-r)
            ddf = tempGauss.fbis(r)
            ddfzero = tempGauss.fbis(0)
            ddfminus = tempGauss.fbis(-r)
            dddf = tempGauss.ftris(r)
            dddfminus = tempGauss.ftris(-r)
            temp[j, i] = 3./2. * u/(r*r*r) * (-3 * dfminus + 3 * df + r * (-8 * ddf + ddf + ddfminus + r * dddfminus - r * dddf)) if r != 0 else 1e+12 
    return np.reshape(temp, -1)


def main():
    NR = 5
    NU = 6

    # Now let's head into the problem, define grids and matrices        
    # Use Derivatives.py and cheb.py to get grid and matrices
    # Notice that we should mirror the X grid for better accuracy -- to do later
    X = Ch.Grid(NR, mode='cheb').grid
    U = Ch.Grid(NU, mode='cheb').grid
    DX = Div.DR(NR).matrix
    DU = Div.DR(NU).matrix

    # costh is just U, sinth is just sqrt(1-u^2)
    sinth = [np.sqrt(1 - u*u) for u in U]

    # same for sinthsquared
    sinthsqr = np.diag([(1 - u*u) for u in U])

    # We also need X matrix, defined as 1/Pi Sin(Pi X)
    sinPix = np.diag([(1/np.pi) * np.sin(np.pi * x) for x in X])
    
    # And a "frac" matrix: 
    frac = np.diag([2*u/(1-u*u) if (1-u*u) != 0 else 1e+12 for u in U])

    # Now all we need is a matrix and RHS vectors and we can the apply a solver, i will assume the convention (U, X), if it does not work one can change i<->j in Kru and eta 
    IdX = np.eye(NR)
    IdU = np.eye(NU)
    eta = Eta(X, U)
    # U = np.diag(U)
    #X = np.diag(X)

    # upper left matrix -> Du + Du*eta
    M11 = np.kron(DU,IdX) + np.dot(np.kron(DU,IdX), eta)

    # upper right matrix -> Du - Du eta - 2u/(1-u^2)
    M12 = np.kron(DU, IdX) - np.dot(np.kron(DU, IdX), eta) - 2*np.kron(frac, IdX) 

    # lower left matrix -> 1/pi sin(pi x) Dx + 3 - 1/2Pi sin(x pi)
    M21 = np.kron(IdU, sinPix @ DX) + 3*np.kron(IdU,IdX) - 1./2.* np.kron(IdU, sinPix)

    # lower right matrix -> 1/(2 pi) Sin(pi x) Dx eta
    M22 = 1./2.* np.diag(np.dot(np.kron(IdU, sinPix @ DX), eta))

    # Now we should join this together: 
    M = np.block([[M11,M12],[M21,M22]])
    
    # And now it is time for the RHS 
    # Upper part -> (1/pi sin(pi x) Dx + 3).K^r_u
    # RHS1 = np.dot(np.kron(IdU @ sinPix, DX) + 3 * np.kron(IdU,IdX) - 1./2. * np.kron(IdU, sinPix), Kru(X,U))
    RHS1 = np.dot(np.kron(IdU,sinPix @ DX) + 3 * np.kron(IdU,IdX), Kru(X,U))

    # Lower part -> ((1 - u^2) - 2 * u).K^r_u
    print("Kru shape: ", Kru(X,U).shape)
    print("rest: ",(np.kron(sinthsqr, IdX) - 2 * np.kron(np.diag(U), IdX)).shape)
    RHS2 = np.dot(np.kron(sinthsqr, IdX) - 2 * np.kron(np.diag(U), IdX), Kru(X, U))
    
    #Total R = [R1, R2]:
    R = np.zeros(2*(NR*NU,))
    R = np.concatenate((RHS1,RHS2))

    # Now we need to solve M.X = R

    X = np.zeros(2*(NR*NU,))
    X = np.linalg.solve(M,R)

    print("Rownanie: A = ", M) 
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Wyraz wolny: B = ", R) 
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Rozwiazanie: X = ", X) 

if __name__== "__main__":
    main()
