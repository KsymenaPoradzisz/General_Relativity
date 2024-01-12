import Derivatives as Div
import Cheb as Ch
import numpy as np
import Data
import Plotting
import matplotlib.pyplot as plt

def main():    
    # Now let's head into the problem, define grids and matrices        
    # Use Derivatives.py and cheb.py to get grid and matrices
    NR, NU = 62, 50
    dimR = (NR-2)//2
    X = Ch.Grid(NR, mode='cheb').grid[1:NR//2]
    U = Ch.Grid(NU, mode='cheb').grid
    DX = Div.DR(NR).matrix[1:-1, 1:-1]
    DU = Div.DR(NU).matrix

    # costh is just U, sinTh is just sqrt(1-u^2)
    sinTh = [np.sqrt(1 - u*u) for u in U]

    # same for sinThSquared
    sinThSqr = np.diag([(1 - u*u) for u in U])

    # We also need X matrix, defined as 1/Pi Sin(Pi X)
    sinPix = np.diag([(1/np.pi) * np.sin(np.pi * x) for x in X])
    sinPixDX = np.kron(np.eye(2), sinPix) @ DX
    sinPixDX = Div.blockSymmetrize(sinPixDX, 'L', NU)

    # And a "frac" matrix: 
    Ufrac = np.diag([2*u/(1-u*u) if (1-u*u) != 0 else 1e+12 for u in U])

    # Now all we need is a matrix and RHS vectors and we can the apply a solver, assuming we work in (X, U) space, if it does not work one can change i<->j in Kru and eta 
    IDx = np.eye(dimR)
    IDu = np.eye(NU)
    eta = Data.Eta(X, U).eta
    kru = Data.Kru(X, U).kru

    # upper left block -> Du + Du*eta
    M11 = np.kron(IDx, DU) + np.diag(np.kron(IDx, DU) @ eta)

    # upper right block -> Du - Du eta - 2u/(1-u^2)
    M12 = np.kron(IDx, DU) - np.diag(np.kron(IDx, DU) @ eta) - np.kron(IDx, Ufrac) 

    # lower left block -> 1/pi sin(pi x) Dx + 3 - 1/2Pi sin(x pi)
    M21 = sinPixDX + 3*np.kron(IDx, IDu) - 1./2.* np.kron(sinPix, IDu)

    # lower right block -> 1/(2 pi) Sin(pi x) * (Dx eta)
    M22 = 1./2.* np.diag(sinPixDX @ eta)

    # a = Div.blockSymmetrize(DX, "L", NU)      # check whether M22 formula is in agreement in more rigorous approach
    # b = (a @ eta)
    # c = np.kron(sinPix, np.eye(NU))
    # M22Prime = 1/2 * np.diag(c @ b)
    # print(M22 - M22Prime)

    # Now we should join this together: 
    operator = np.block([[M11,M12],[M21,M22]])
    
    # And now it is time for the RHS 
    # Upper part -> (1/pi sin(pi x) Dx + 3).K^r_u
    # RHS1 = np.dot(np.kron(IDu @ sinPix, DX) + 3 * np.kron(IDu,IDx) - 1./2. * np.kron(IDu, sinPix), Kru(X,U))
    RHS1 = np.dot(sinPixDX + 3 * np.kron(IDx, IDu) - 1./2. * np.kron(sinPix, IDu), kru)

    # Lower part -> ((1 - u^2) - 2 * u).K^r_u
    print("Kru shape: ", kru.shape)
    print("rest: ",(np.kron(IDx, sinThSqr) - 2 * np.kron(IDx, np.diag(U))).shape)
    RHS2 = np.dot(np.kron(IDx, sinThSqr) - 2 * np.kron(IDx, np.diag(U)), kru)
    
    #Total R = [R1, R2]:
    source = np.concatenate((RHS1, RHS2))

    # Now we need to solve linear matrix equation 
    x = np.linalg.solve(operator, source)

    # print("Rownanie: A = ", operator) 
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print("Wyraz wolny: B = ", source) 
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print("Rozwiazanie: X = ", x) 

    Krr = x[:len(x)//2]
    Ktt = x[len(x)//2:]

    # Plotting.PolarPlotter().plot(X, U, Krr)
    Krr = Krr.reshape((X.shape[0], U.shape[0]))
    Ktt = Ktt.reshape((X.shape[0], U.shape[0]))
    plt.plot(X, Ktt[:, 20])
    plt.show()

if __name__== "__main__":
    main()
