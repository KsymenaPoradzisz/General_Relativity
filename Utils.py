import Derivatives as Div
from Cheb import Grid
import numpy as np
import Data

# ------------------------------------- CONVERSIONS ---------------------------------------------------------

# if argument is a Grid instance then get its 'grid' property which is a list
def gridToArray(grid: Grid) -> list:
    if isinstance(grid, Grid):
        return grid.grid


# ------------------------------------ MATRIX UTILITIES -----------------------------------------------------

# get rid of the first and the last row and collumn (i.e. for imposing boundary conditions)
def stripMatrix(matrix: list[list], first: bool = True, last: bool = True) -> list[list]:
    if last:
        return matrix[first:-1, first:-1]
    else:
        return matrix[first:, first: ]

# ----------------------------------- USEFUL DATA -----------------------------------------------------------

# sin(x)
def sinTh(gridU: Grid|list) -> list:
    # make sure gridX is actually an array
    if gridU.gridType != "cheb": print("Grid is not 'cheb' type. Are you sure it is intended?")
    gridU = gridToArray(gridU)

    return [np.sqrt(1 - u*u) for u in gridU]


# sin^2(x)
def sinThSq(gridU: Grid|list) -> list:
    # make sure gridX is actually an array
    if gridU.gridType != "cheb": print("Grid is not 'cheb' type. Are you sure it is intended?")    
    gridToArray(gridU)

    return [(1 - u*u) for u in gridU]


# 1/pi * sin(pi * x)
def sinPiX(gridX: Grid|list) -> list:
    # make sure gridX is actually an array
    if gridX.gridType != "cheb": print("Grid is not 'cheb' type. Are you sure it is intended?")
    gridToArray(gridX)

    return [(1/np.pi) * np.sin(np.pi * x) for x in gridX]


# --------------------------------------- MATH OPERATIONS ----------------------------------------------------

def gridDotMatrix(gridX: Grid, derivative: Div.Derivative) -> list[list]:
    arr = gridX.halfGrid()
    div = derivative.matrix

    # if shapes match then just multiply matrices:     [[grid, 0], [0, grid]] @ Div
    if len(arr) == div.shape[0] // 2:
        return np.kron(np.eye(2), arr) @ div
    
    # if shapes don't match it means that matrix is a full matrix and gridX rejects boundary point
    elif len(arr) == (div.shape[0] - 2) // 2:
        return np.kron(np.eye(2), arr) @ stripMatrix(div)
    
    elif len(arr) - 1 == div.shape[0] // 2:
        return np.kron(np.eye(2), gridX.halfGridNoBoundary()) @ div
    
    else: 
        raise ValueError(f"Object shapes do not match: grid: {len(arr)}, div: {(div.shape)}")