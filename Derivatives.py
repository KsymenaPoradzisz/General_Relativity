import numpy as np
import Cheb as Ch
import Data
from scipy.linalg import toeplitz

# storage for derivative matrices
class Derivative:
    matrix = None  # static variable to be easily accessible without redundant copying

    def __repr__(self) -> str:
        return f"{np.around(self.matrix, 4)}"


class DR(Derivative):
    # use for unevenly spaced Chebyshev grids (radial for example)
    def __init__(self, N) -> None:
        # Chebyshev polynomial differentiation matrix.
        # Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
        
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
        dr = DR(N).matrix
        self.matrix = dr @ dr


class DTheta(Derivative):
    # use for periodic, regular grid (angular for example)
    def __init__(self, N: int) -> None:
        # Differentiation matrix for uniform grid
        # Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
        h = 2*np.pi / N
        arr = ((-1.0)**np.arange(1, N))
        col = np.zeros(N)
        col[1:] = 0.5 * arr / np.tan(np.arange(1, N) * h/2.0)
        row = np.zeros(N)
        row[0] = col[0]
        row[1:] = col[N-1:0:-1]

        self.matrix = toeplitz(col, row)


class DTheta2(Derivative):
    # use for periodic, regular grid (angular for example)
    def __init__(self, N: int) -> None:
        dTheta = DTheta(N).matrix
        self.matrix = dTheta @ dTheta


class DX(Derivative):
    # use for nonperiodic, regular grid (currently not needed)
    pass



def blockSymmetrize(matrix, divPosition: str, sizeId: int) -> list[list]:
    # divPosition refers to the position of multiplied matrix - it is either matrix x Id ('L' mode) or Id x matrix ('R' mode)
    # sizeId is an integer size of target identity matrix
    dim = matrix.shape[0]
    blockLU, blockRU = matrix[:dim//2, :dim//2], matrix[:dim//2, dim//2:]

    diagId = Data.DiagId(sizeId).matrix
    antiDiagId = Data.AntiDiagId(sizeId).matrix
    match divPosition:
        case 'L':
            return np.kron(blockLU, diagId) + np.kron(blockRU, antiDiagId)
        case 'R':
            return np.kron(diagId, blockLU) + np.kron(antiDiagId, blockRU)
        case _:
            raise ValueError("'divPosition' should be 'L' or 'R'!")