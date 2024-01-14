import numpy as np
from Cheb import Grid
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
        x = Grid(N, mode='cheb').grid
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
    def __init__(self, N: int) -> None:
        pass


class Laplacian(Derivative):
    def __init__(self, gridR: Grid, gridTheta: Grid, gridPhi: Grid = None, mode="spherical", isGridRCompactified: bool = False) -> None:
        # available modes: "spherical" (3D), "polar2D",  "cartesian3D", "cartesian2D"
        # modes "spherical", "polar3D" and "polar2D" will be used most extensively, hence the naming
        match mode:
            case "spherical":
                self.matrix = self.spherical(gridR, gridTheta, gridPhi, isGridRCompactified)
            case "polar2D":
                self.matrix = self.polar2D(gridR, gridTheta, isGridRCompactified)
            case "cartesian2D":
                self.matrix = self.cartesian2D(gridR, gridTheta)
            case _:
                ValueError('Available modes: "spherical" (3D), "polar" (2D), "cartesian2D".')

    def spherical(self, gridR: Grid, gridU: Grid, gridPhi: Grid, isGridRCompactified: bool):
        # assuming cylindrical symmetry as it is the case in the project
        # thus the operator matrix will be of size (NR*NTheta) x (NR*NTheta)

        # x = array of compactified distance r: r = 2/pi * tan(pi/2 * x)
        # u = array of compactified spherical angle theta: u = cos(Theta)
        NR, NTheta = len(gridR.grid), len(gridU.grid)

        # case when gridR represents X - compactified Chebyshev spatial variable, else - space is restricted to a disc of radius=1
        if isGridRCompactified:
            r = 2/np.pi * np.tan(np.pi / 2 * gridR.grid)                    # r = 2/pi * tan(pi/2 * x) WARNING if L =! 2/Pi
            dr = np.diag(1/np.cos(np.pi/2 * gridR.grid)**2) @ DR(NR).matrix   # dr = 1 / cos^2(pi/2 * x) dx
            rInverse = np.diag(1/r[1:NR//2])                                # diagonal matrix of 1/r for positive distances r
        else: 
            r = gridR.grid
            dr = DR(NR).matrix

        # case when gridU represents U - compactified Chebyshev angular variable, else - uniform grid
        if gridU.gridType == "cheb":
            theta = np.array([np.arccos(u) for u in gridU.grid])
            dTheta = np.diag(-1*np.sqrt(np.ones(NTheta) - gridU.grid**2)) @ DR(NTheta).matrix   #dTheta = -sqrt(1-u^2) du
        else:
            theta = gridU.grid
            dTheta = DTheta(NTheta).matrix


        # get only positive part of radial grid
        rInverse = np.diag(1 / r[:NR//2])
        rInvDr = (np.kron(np.identity(2), rInverse) @ dr)   # 1/R * DR
        rInverse = rInverse[1:, 1:]                         # get rid of boundary r=0 radial point

        # 1/r**2 * ctg(Theta) * DTheta
        CtgThetaPart = np.kron(rInverse @ rInverse, np.diag(1./np.tan(theta)) @ dTheta)
        
        # 1/r DR
        rInverseDrPart = blockSymmetrize(rInvDr, NTheta)

        return self.polar2D(gridR, gridU, isGridRCompactified) + CtgThetaPart + rInverseDrPart
        

    def polar2D(self, gridR: Grid, gridTheta: Grid, isGridRCompactified):
        NR, NTheta = gridR.size, gridTheta.size

        # case when gridR represents X - compactified Chebyshev spatial variable, else - space is restricted to a disc of radius=1
        if isGridRCompactified:
            r = 2/np.pi * np.tan(np.pi / 2 * gridR.grid)                    # r = 2/pi * tan(pi/2 * x)
            dr = np.diag(np.cos(np.pi/2 * gridR.grid)**2) @ DR(NR).matrix   # dr = cos^2(pi/2 * x) dx
        else: 
            r = gridR.grid
            dr = DR(NR).matrix

        # case when gridTheta represents U - compactified Chebyshev angular variable, else - uniform grid
        if gridTheta.gridType == "cheb":
            dTheta = np.diag(-1*np.sqrt(np.ones(NTheta) - gridTheta.grid**2)) @ DR(NTheta).matrix   #dTheta = -sqrt(1-u^2) du
        else:
            dTheta = DTheta(NTheta).matrix

        # get only positive part of radial grid
        rInverse = np.diag(1 / r[:NR//2])
        rInvDr = np.kron(np.identity(2), rInverse) @ dr     # 1/R * DR
        rInverse = rInverse[1:, 1:]                         # get rid of boundary r=0 radial point

        # create each of 2D polar laplacian compoment
        DrSqPart = blockSymmetrize(dr @ dr, NTheta)
        rInverseDrPart = blockSymmetrize(rInvDr, NTheta)        
        rInverseSqDThetaSq = np.kron(rInverse @ rInverse, dTheta @ dTheta)

        return  rInverseDrPart + DrSqPart + rInverseSqDThetaSq


    def cartesian2D(self, gridR: Grid, gridTheta: Grid):
        if gridR.gridType == "cheb": DX = DR(gridR.size).matrix
        else: TypeError("Currently cartesian DX derivatives are supported only on Chebyshev grids.")

        if gridTheta.gridType == "cheb": DY = DR(gridTheta.size).matrix
        else: TypeError("Currently cartesian DY derivatives are supported only on Chebyshev grids.")

        # trim first nad last row nad column of matrices -> imposing Dirichlet zero boundary conditions
        dimX, dimY = DX.shape[0]-2, DY.shape[0]-2
        dxSquared = (DX @ DX)[1:-1, 1:-1]
        dySquared = (DY @ DY)[1:-1, 1:-1]

        return np.kron(np.identity(dimX), dySquared) + np.kron(dxSquared, np.identity(dimY))


def blockSymmetrize(matrix, sizeId: int) -> list[list]:
    # divPosition refers to the position of multiplied matrix - it is either matrix x Id ('L' mode) or Id x matrix ('R' mode)
    # sizeId is an integer size of target identity matrix
    dim = matrix.shape[0]
    blockLU, blockRU = matrix[1:dim//2, 1:dim//2], matrix[1:dim//2, dim//2:-1]

    diagId = Data.DiagId(sizeId).matrix
    antiDiagId = Data.AntiDiagId(sizeId).matrix

    return np.kron(blockLU, diagId) + np.kron(blockRU, antiDiagId)
