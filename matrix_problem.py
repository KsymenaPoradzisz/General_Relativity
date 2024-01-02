from Derivatives.py import *
import numpy as np
import math
import argparse # this library is for us to parse arguments via console

class Gauss:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma - sigma

    def f(self,x):
        e = math.e 
        return e**((x-mu)**2/sigma**2)
    def fprim(self,x):
        e = math.e 
        return e**((x-mu)**2/sigma**2) * (2*(x-mu))/sigma**2
    def fbis(self,x):
        e = math.e
        return e**((x-mu)**2/sigma**2) * (2*(x-mu))/sigma**2
#Initial condition for eta
def eta(epsilon, t, r, u):
    return -3/2


def main():
    
    parser = argparse.ArgumentParser(description="Solve Professors problem")
    
    parser = argparse.add_argument("-nr", "--numbergridR", type=int, help="this number defines dim of grid for radial variable") 
    parser = argparse.add_argument("-nt", "--numbergridtheta", type=int, help="th    is number defines dim of grid for angle variable")

# For code to work, one must type for example python3 matrix_problem.py -nr 81 -nt 12 or --numbergridR 54 --numbergridtheta 9
    

    args = parser.parse_args()

#Some security for it to work

    if args.numbergridR is not None and args.numbergridtheta is not None:
        nr = args.numbergridR
        nt = args.numbergridtheta

#Now let's head into the problem, define grids and matrices        
    gridR = np.empty((nr,))
    gridT = np.empty((nt,))

    CMatrixR = np.empty((nr,nr))
    CmatrixT = np.empty((nt,nt))
#Use cheb(N) method from Derivatives.py
    RR = Derivatives()
    CMatrixR,gridR = RR(nr)
    CmatrixT,gridT = RR(nt)

#costh is just gridT
    costh = [] 
    costh = [u for u in gridT]

#sinth is just sqrt(1-u^2)
    sinth = []
    sinth = [(1-u*u)**(1/2) for u in gridT]
#same for sinthsquared
    sinthsqr = []
    sinthsqr = [(1-u*u) for u in gridT]
#Initial condition for eta

if __name__== "__main__":
    main()
