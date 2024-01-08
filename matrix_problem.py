import Derivatives as Div
import Cheb as Ch
import numpy as np
import math

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
    NR = 6
    NU = 6

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
    sinthsqr = np.diag(sinthsqr)

#We also need X matrix, defined as 1/Pi Sin(Pi X)
    
    sinPix = []
    sinPix = [(1/np.pi) * np.sin(np.pi * x) for x in X]
    sinPix = np.diag(sinPix)
    
#And a "frac" matrix: 
    
    frac = [] 
    frac = [2*u/(1-u*u) if (1-u*u) != 0 else 1e+12 for u in U]
    frac = np.diag(frac)
#Initial condition for eta -- we need to define it, it needs 2 grids and 3 functions, but gaussians
#are already defined upstairs, so i will use them (this is wrong because not compactified!!!)

    def Eta(X, U):
        NX = len(X)
        NU = len(U)
        temp = np.zeros((NX, NU))
        for i in range(NX):
            for j in range(NU):
                u = U[j]
                x = X[i]
                r = 2./np.pi * np.tan(np.pi * x / 2)
                tempGauss = Gauss(0,0.1) #example of gauss parameters
                f = tempGauss.f(r)
                df = tempGauss.fprim(r)
                dfminus = tempGauss.fprim(-r)
                dfzero = tempGauss.fprim(0.)
                ddf = tempGauss.fbis(r)
                ddfminus = tempGauss.fbis(-r)
                print("u = ", u)
                print("r = ", r)
                print("dfzero = ", dfzero)
                print("dfminus = ", dfminus)
                print("df = ", df)
                print("f = ", f)
                print("ddf = ", ddf)
                print("ddfminus = ", ddfminus)
                temp[i,j] = -3/2 * (u * u - 1) / (r * r) *(-4. * dfzero + 2. * dfminus + 2. * df + r * ddfminus - r * ddf) 

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
                r = 2./np.pi * np.tan(np.pi * x / 2)
                tempGauss = Gauss(0,0.1) #example of gauss parameters
                df = tempGauss.fprim(r)
                dfminus = tempGauss.fprim(-r)
                ddf = tempGauss.fbis(r)
                ddfzero = tempGauss.fbis(0)
                ddfminus = tempGauss.fbis(-r)
                dddf = tempGauss.ftris(r)
                dddfminus = tempGauss.ftris(-r)
                temp[i,j] = 3./2. * u/(r*r*r) * (-3 * dfminus + 3 * df + r * (-8 * ddf + ddf + ddfminus + r * dddfminus - r * dddf))
        return np.reshape(temp,-1)

#Now all we need is a matrix and RHS vectors and we can the apply a solver, i will assume the convention (U, X), if it does not work one can change i<->j in Kru and eta 

    IdX = np.eye(NR)
    IdU = np.eye(NU)
    eta = Eta(X,U)
    U = np.diag(U)
    #X = np.diag(X)
#upper left matrix -> Du + Du*eta
    
    M11 = np.kron(DU,IdX) + np.dot(np.kron(DU,IdX), eta)
    print("M11 ", np.shape(M11))
#upper right matrix -> Du - Du eta - 2u/(1-u^2)
    
    #print("Wymiar M12 -> ", np.shape(M12), " Oraz wymiar eta -> ", np.shape(eta), " Oraz wymiar iloczynu -> ", np.shape(np.dot(np.kron(IdX,DU),eta)), " Oraz wymiar frac -> ", np.shape(np.kron(IdX,frac)))
    M12 = np.kron(DU,IdX) - np.dot(np.kron(DU,IdX), eta) - 2*np.kron(frac,IdX) # tu chyba error
    print("M12 ", np.shape(M12))
#lower left matrix -> 1/pi sin(pi x) Dx + 3 - 1/2Pi sin(x pi)

    M21 = np.kron(IdU,sinPix @ DX) + 3 * np.kron(IdU,IdX) -1./2. * np.kron(IdU,sinPix)
    print("M21 ", np.shape(M21))
#lower right matrix -> 1/(2 pi) Sin(pi x) Dx eta

    M22 = 1./2.* np.diag(np.dot(np.kron(IdU, sinPix @ DX), eta))
    print("M22 ", np.shape(M22))

#Now we should join this together: 

    M = np.block([[M11,M12],[M21,M22]])

#And now it is time for the RHS 

#Upper part -> (1/pi sin(pi x) Dx + 3).K^r_u

    RHS1 = np.dot(np.kron(IdU,sinPix @ DX) + 3 * np.kron(IdU,IdX) - 1./2. * np.kron(IdU, sinPix), Kru(X,U))

#Lower part -> ((1 - u^2) - 2 * u).K^r_u
    print("Kru shape ", Kru(X,U).shape)
    print("rest ",(np.kron(sinthsqr,IdX) - 2 * np.kron(U,IdX)).shape)
    RHS2 = np.dot(np.kron(sinthsqr,IdX) - 2 * np.kron(U,IdX), Kru(X,U))


#Now we could test but i am afraid to type python3 matrixproblem.py because it will probably have a lot of errors xd
    
if __name__== "__main__":
    main()
