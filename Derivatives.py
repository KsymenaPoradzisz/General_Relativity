import numpy as np
import math


# storage for derivative matrices
class Derivative:
    _matrix = None  # static variable to be easily accessible without redundant copying

    def __init__(self, size: list) -> None:
        self._matrix = np.zeros(size)
        
    def cheb(N):
    '''Chebyshev polynomial differentiation matrix.
       Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    '''
    x      = cos(pi*arange(0,N+1)/N)
    if N%2 == 0:
        x[N//2] = 0.0 # only when N is even!
    c      = ones(N+1); c[0] = 2.0; c[N] = 2.0
    c      = c * (-1.0)**arange(0,N+1)
    c      = c.reshape(N+1,1)
    X      = tile(x.reshape(N+1,1), (1,N+1))
    dX     = X - X.T
    D      = dot(c, 1.0/c.T) / (dX+eye(N+1))
    D      = D - diag( D.sum(axis=1) )
    return D,x
 


class DR(Derivative):
    def __init__(self, size: list) -> None:
        super().__init__(size)



class DTheta(Derivative):
    def __init__(self, size: list) -> None:
        super().__init__(size)  
         
