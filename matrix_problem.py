from Derivatives import *
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
        return np.exp((x - self.mu)**2 / self.sigma**2) * (2 *(x - self.mu)) / self.sigma**2
    
#Initial condition for eta
def eta(epsilon, t, r, u):
    return -3/2


def main():
    NR = 40
    NTheta = 20

#Now let's head into the problem, define grids and matrices        
    gridR = np.empty((NR,))
    gridT = np.empty((NTheta,))

    CMatrixR = np.empty((NR,NR))
    CmatrixT = np.empty((NTheta, NTheta))

#Use cheb(N) method from Derivatives.py
    RR = Derivative()
    CMatrixR,gridR = RR(NR)
    CmatrixT,gridT = RR(NTheta)

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
