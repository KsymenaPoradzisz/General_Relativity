import numpy as np
import Cheb as Ch
import Derivatives as Div
from scipy.integrate import RK45

class Solver:
    pass


class SpectralSolverCart2D(Solver):
    # assumptions:
    # - zero valued boundary conditions (neglecting first and last element of grid)
    # - operator is a matrix of size (NX-2 * NY-2)**2
    # - source is a vector of size (NX-2 * NY-2)
    # - returns vector of size (NX-2 * NY-2)
    def solve(self, operator: list[list], source: list) -> None:

        return np.linalg.solve(operator, source).reshape(-1)


class SpectralSolverPolar2D(Solver):
    def solve(self, operator):
        pass



class TimeSolver(Solver):
    def timestep(function, t0, y0, returnsVector: bool = True):
        pass
