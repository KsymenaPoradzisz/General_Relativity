import numpy as np
import Cheb as Ch

# storage for derivative matrices
class Derivative:
    matrix = None  # static variable to be easily accessible without redundant copying

    def __init__(self, size) -> None:
        self.matrix = np.zeros(size)
        self.matrix = self.cheb(size)[0] 

    def __repr__(self) -> str:
        return f"{np.around(self.matrix, 4)}"


class DX(Derivative):
    def __init__(self, N) -> None:
        # Chebyshev polynomial differentiation matrix.
        #Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
        
        # x - N dimensional array of Chebyshev points
        # D - Matrix (N)x(N) dimensional (for spectral differentiation I guess?) 

        # UGLY AF - TO REWRITE
        x = Ch.Grid(N).grid
        c = np.ones(N, dtype=np.float64)
        c[0] = c[-1] = 2.0
        c *= (-1.0)**np.arange(0, N)
        c = c.reshape(N, 1)
        X = np.tile(x.reshape(N, 1), (1, N))
        dX = X - X.T
        DivMatrix = np.dot(c, 1.0 / c.T) / (dX + np.eye(N))
        DivMatrix -= np.diag( DivMatrix.sum(axis=1) )

        self.matrix = DivMatrix


class DR(Derivative):
    def __init__(self, size: list) -> None:
        super().__init__(size)



class DTheta(Derivative):
    def __init__(self, size: list) -> None:
        super().__init__(size)  
         
