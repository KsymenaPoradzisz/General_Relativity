import numpy as np

class Grid:
    grid = None

    def __init__(self, N, mode="cheb") -> None:
        # allowed modes: "uni" for uniform and "cheb" for Chebyshev grid
        match mode:
            case "uni":
                self.grid = 2*np.array([i/(N-1) for i in range(N)], dtype=np.float64) - 1
            case "cheb":
                # Chebyshev polynomial differentiation matrix.
                #Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
                
                # x - N dimensional array of Chebyshev points
                self.grid = np.array(np.cos(np.pi * np.arange(0, N) / (N-1)), dtype=np.float64)
                if N % 2 == 1:
                    self.grid[N//2] = 0.0   # only for N odd!
            case _:
                raise ValueError("Choose 'uni' or 'cheb' mode!")


                