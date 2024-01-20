import Derivatives as Div
from Cheb import Grid
from Utils import (sinTh, sinThSq, sinPiX, fracU, gridDotMatrix, 
                   halfGrid, halfGridNoBoundary)
from Data import Gauss, Eta, Kru
import numpy as np
import Plotting
import matplotlib.pyplot as plt
from tests import *

def main():    
    # Now let's head into the problem, define grids and matrices
    NR, NU = 42, 30
    dimR = (NR-2)//2
    X = Grid(NR, 'cheb')
    U = Grid(NU, 'cheb')
    DX = Div.DR(NR).matrix
    DU = Div.DR(NU).matrix

    # costh is just U, sinTh is just sqrt(1-u^2), will also use sinThSquared and a "frac" matrix: 
    sinThSqr = np.diag(sinThSq(U))
    Ufrac = np.diag(fracU(U))

    # We also need an array of 1/Pi Sin(Pi X)
    sinPixDX = gridDotMatrix(sinPiX(X), DX)
    sinPixDX = Div.blockSymmetrize(sinPixDX, NU)
    sinPix = np.diag( sinPiX( halfGridNoBoundary(X) ) )

    # Now all we need is operator and source and we can the apply a solver. 
    # Assume we work in (X, U) space
   
    # Compactified variable is inside tan(pi X/2) * L, so we need to include that in gauss
    L = 2/np.pi
    # X = [L * np.tan(x * np.pi/2) if (x != 0 and x != 1) else (1e+5 if x == 0 else -1e+5) for x in X]
    X = X.grid[1:NR//2]
    U = U.grid
    IDx = np.eye(dimR)
    IDu = np.eye(NU)
    eta = Eta(X, U).init_cond("gauss").eta
    kru = Kru(X, U).init_cond("gauss").kru

    # upper left block -> Du + Du*eta
    # upper right block -> Du - Du eta - 2u/(1-u^2)
    # lower left block -> 1/pi sin(pi x) Dx + 3 - 1/2Pi sin(x pi)
    # lower right block -> 1/(2 pi) Sin(pi x) * (Dx eta)
    M11 = np.kron(IDx, DU) + np.diag(np.kron(IDx, DU) @ eta)
    M12 = np.kron(IDx, DU) - np.diag(np.kron(IDx, DU) @ eta) - np.kron(IDx, Ufrac) 
    M21 = sinPixDX + 3*np.kron(IDx, IDu) - 1./2.* np.kron(sinPix, IDu)
    M22 = 1./2.* np.diag(sinPixDX @ eta)

    # Now we should join this together: 
    operator = np.block([[M11,M12],[M21,M22]])

    # And now it is time for the RHS 
    # Upper part -> (1/pi sin(pi x) Dx + 3).K^r_u
    # Lower part -> ((1 - u^2) - 2 * u).K^r_u
    RHS1 = np.dot(sinPixDX + 3 * np.kron(IDx, IDu) - 1./2. * np.kron(sinPix, IDu), kru)
    RHS2 = np.dot(np.kron(IDx, sinThSqr) - 2 * np.kron(IDx, np.diag(U)), kru)

    # Total R = [R1, R2]:
    source = np.concatenate((RHS1, RHS2))

    # Now we need to solve linear matrix equation 
    x = np.linalg.solve(operator, source)
    Krr = x[:len(x)//2]
    Ktt = x[len(x)//2:]


    # Plot results
    Krr = Krr.reshape((-1, NU))
    Ktt = Ktt.reshape((-1, NU))
    plt.plot(X, Krr[:, 20])

    temp = Tests(sigma = 0.5, epsilon = 0.000001, mode = 'Krr')

    testKrr = np.zeros((len(X), NU)) 
    for i in range(len(X)):
        for j in range(NU):
            x, u = X[i], U[j]
            testKrr[i,j] = temp.Linear(x, u, mode= 'Krr')
    plt.plot(X, testKrr[:,20])
    plt.show()


if __name__== "__main__":
    main()
