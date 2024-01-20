import Derivatives as Div
from Cheb import Grid
import numpy as np

# ------------------------------------- CONVERSIONS ---------------------------------------------------------

# if argument is a Grid instance then get its 'grid' property which is a list
def gridToArray(grid: Grid) -> list:
    if isinstance(grid, Grid):
        return grid.grid
    return grid


def halfGrid(grid):
    grid = gridToArray(grid)
    return grid[ : len(grid)//2]


def halfGridNoBoundary(grid):
    grid = gridToArray(grid)
    return grid[1 : len(grid)//2]


# ------------------------------------ MATRIX UTILITIES -----------------------------------------------------

# get rid of the first and the last row and collumn (i.e. for imposing boundary conditions)
def strip(matrix: list[list], first: bool = True, last: bool = True) -> list[list]:
    if last:
        return matrix[first:-1, first:-1]
    else:
        return matrix[first:, first: ]

# ----------------------------------- USEFUL DATA -----------------------------------------------------------

# sin(x)
def sinTh(gridU: Grid|list) -> list:
    # make sure gridX is actually an array
    # if gridU.gridType != "cheb": print("\nGrid is not 'cheb' type. Are you sure it is intended?\n")
    gridU = gridToArray(gridU)

    return np.array([np.sqrt(1 - u*u) for u in gridU])


# sin^2(x)
def sinThSq(gridU: Grid|list) -> list:
    # make sure gridX is actually an array
    # if gridU.gridType != "cheb": print("\nGrid is not 'cheb' type. Are you sure it is intended?\n")    
    gridU = gridToArray(gridU)

    return np.array([(1 - u*u) for u in gridU])


# 1/pi * sin(pi * x)
def sinPiX(gridX: Grid|list) -> list:
    # make sure gridX is actually an array
    # if gridX.gridType != "cheb": print("\nGrid is not 'cheb' type. Are you sure it is intended?\n")
    gridX = gridToArray(gridX)

    return np.array([(1/np.pi) * np.sin(np.pi * x) for x in gridX])


def fracU(gridU: Grid|list):
    gridU = gridToArray(gridU)

    return np.array([2*u/(1-u*u) if (1-u*u) != 0 else 1e+12 for u in gridU])


# --------------------------------------- MATH OPERATIONS ----------------------------------------------------

def gridDotMatrix(gridX: Grid|list, derivative: list[list]) -> list[list]:
    # handle many multiplication cases for grid and derivative dimensions 
    arr = halfGrid(gridX)
    arrBoundless = halfGridNoBoundary(gridX)
    try: div = derivative.matrix
    except: div = derivative

    # if shapes match then just multiply matrices:     [[grid, 0], [0, grid]] @ Div
    if 2*len(arr) == div.shape[0]:
        return np.kron(np.eye(2), np.diag(arr)) @ div 
    
    # if shapes don't match it means that matrix is a full matrix and gridX rejects boundary point
    # or matrix is striped and grid contains boundary point(s)
    elif 2*len(arr) == (div.shape[0] - 2):
        return np.kron(np.eye(2), np.diag(arr)) @ strip(div)
    
    elif 2*len(arr) - 2 == div.shape[0]:
        return np.kron(np.eye(2), np.diag(arrBoundless)) @ div
    
    else: 
        raise ValueError(f"Object shapes do not match: grid: {len(arr)}, div: {(div.shape)}")