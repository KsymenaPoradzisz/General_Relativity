import Derivatives as Div
import Cheb as Ch
import numpy as np
import Data


def main():    
    # Now let's head into the problem, define grids and matrices        
    # Use Derivatives.py and cheb.py to get grid and matrices
    NR, NU = 6, 6
    dimR = (NR-2)//2
    X = Ch.Grid(NR, mode='cheb').grid[1:NR//2]
    U = Ch.Grid(NU, mode='cheb').grid
    DX = Div.DR(NR).matrix[1:-1, 1:-1]
    DX = Div.blockSymmetrize(DX, "L", NU)
    DU = Div.DR(NU).matrix[1:-1, 1:-1]

    # costh is just U, sinth is just sqrt(1-u^2)
    sinth = [np.sqrt(1 - u*u) for u in U]

    # same for sinthsquared
    sinthsqr = np.diag([(1 - u*u) for u in U])

    # We also need X matrix, defined as 1/Pi Sin(Pi X)
    sinPix = np.diag([(1/np.pi) * np.sin(np.pi * x) for x in X])
    
    # And a "frac" matrix: 
    frac = np.diag([2*u/(1-u*u) if (1-u*u) != 0 else 1e+12 for u in U])

    # Now all we need is a matrix and RHS vectors and we can the apply a solver, assuming we work in (X, U) space, if it does not work one can change i<->j in Kru and eta 
    IdX = np.eye(dimR)
    IdU = np.eye(NU)
    eta = Data.Eta(X, U).eta
    kru = Data.Kru(X, U).kru

    # upper left block -> Du + Du*eta
    M11 = np.kron(IdX, DU) + np.dot(np.kron(IdX, DU), eta)

    # upper right block -> Du - Du eta - 2u/(1-u^2)
    M12 = np.kron(DU, IdX) - np.dot(np.kron(DU, IdX), eta) - 2*np.kron(frac, IdX) 

    # lower left block -> 1/pi sin(pi x) Dx + 3 - 1/2Pi sin(x pi)
    M21 = np.kron(IdU, sinPix @ DX) + 3*np.kron(IdU,IdX) - 1./2.* np.kron(IdU, sinPix)

    # lower right block -> 1/(2 pi) Sin(pi x) Dx eta
    M22 = 1./2.* np.diag(np.dot(np.kron(IdU, sinPix @ DX), eta))

    # Now we should join this together: 
    operator = np.block([[M11,M12],[M21,M22]])
    
    # And now it is time for the RHS 
    # Upper part -> (1/pi sin(pi x) Dx + 3).K^r_u
    # RHS1 = np.dot(np.kron(IdU @ sinPix, DX) + 3 * np.kron(IdU,IdX) - 1./2. * np.kron(IdU, sinPix), Kru(X,U))
    RHS1 = np.dot(np.kron(IdU,sinPix @ DX) + 3 * np.kron(IdU,IdX), kru)

    # Lower part -> ((1 - u^2) - 2 * u).K^r_u
    print("Kru shape: ", kru.shape)
    print("rest: ",(np.kron(sinthsqr, IdX) - 2 * np.kron(np.diag(U), IdX)).shape)
    RHS2 = np.dot(np.kron(sinthsqr, IdX) - 2 * np.kron(np.diag(U), IdX), kru)
    
    #Total R = [R1, R2]:
    source = np.concatenate((RHS1, RHS2))

    # Now we need to solve linear matrix equation 
    x = np.linalg.solve(operator, source)

    print("Rownanie: A = ", operator) 
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Wyraz wolny: B = ", source) 
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Rozwiazanie: X = ", x) 

if __name__== "__main__":
    main()
