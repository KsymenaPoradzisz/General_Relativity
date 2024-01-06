import numpy as np
import Cheb as Ch
from scipy.linalg import toeplitz

# storage for derivative matrices
class Derivative:
    matrix = None  # static variable to be easily accessible without redundant copying

    def __init__(self, size) -> None:
        self.matrix = np.zeros(size)
        self.matrix = self.cheb(size)[0] 

    def __repr__(self) -> str:
        return f"{np.around(self.matrix, 4)}"


class DR(Derivative):
    # use for unevenly spaced Chebyshev grids (radial for example)
    def __init__(self, N) -> None:
        # Chebyshev polynomial differentiation matrix.
        #Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
        
        # x - N dimensional array of Chebyshev points
        # D - Matrix (N)x(N) dimensional 
        x = Ch.Grid(N, mode='cheb').grid
        c = np.ones(N, dtype=np.float64)
        c[0] = c[-1] = 2.0
        c *= (-1.0)**np.arange(0, N)
        c = c.reshape(N, 1)
        X = np.tile(x.reshape(N, 1), (1, N))
        dX = X - X.T
        DivMatrix = np.dot(c, 1.0 / c.T) / (dX + np.eye(N))
        DivMatrix -= np.diag( DivMatrix.sum(axis=1) )

        self.matrix = DivMatrix


class DR2(Derivative):
    # second order differentiation matrix for unevenly spaced Chebyshev grids
    def __init__(self, N) -> None:
        x = Ch.Grid(N).grid
        c = np.ones(N, dtype=np.float64)
        c[0] = c[-1] = 2.0
        c *= (-1.0)**np.arange(0, N)
        c = c.reshape(N, 1)
        X = np.tile(x.reshape(N, 1), (1, N))
        dX = X - X.T
        DivMatrix = np.dot(c, 1.0 / c.T) / (dX + np.eye(N))
        DivMatrix -= np.diag( DivMatrix.sum(axis=1) )

        self.matrix = DivMatrix @ DivMatrix


class DTheta(Derivative):
    # use for periodic, regular grid (angular for example)
    def __init__(self, N: int, interval_length = 2*np.pi) -> None:
        h = interval_length / N
        arr = ((-1.0)**np.arange(1, N))
        col = np.zeros(N)
        col[1:] = 0.5 * arr / np.tan(np.arange(1, N) * h/2.0)
        row = np.zeros(N)
        row[0] = col[0]
        row[1:] = col[N-1:0:-1]
        # print(col)

        self.matrix = toeplitz(col, row)



class DX(Derivative):
    # use for nonperiodic, regular grid (CURRENTLY NOT IMPLEMENTED / NEEDED)
    def __init__(self, size: list) -> None:
        super().__init__(size)

