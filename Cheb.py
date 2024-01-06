import numpy as np

class Grid:
    grid = None

    def __init__(self, N, mode="cheb") -> None:
        # allowed modes: "uni" for uniform and "cheb" for Chebyshev grid
        match mode:
            case "uni":
                self.grid = 2*np.pi* np.array([i/N for i in range(1, N+1)], dtype=np.float64)
            case "cheb":
                # Chebyshev polynomial differentiation matrix.
                #Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
                
                # x - N dimensional array of Chebyshev points
                self.grid = np.array(np.cos(np.pi * np.arange(0, N) / (N-1)), dtype=np.float64)
                self.grid = np.flip(self.grid) # make grid running from -1 to 1 instead of 1 to -1
                if N % 2 == 1:
                    self.grid[N//2] = 0.0   # only for N odd!
            case _:
                raise ValueError("Choose 'uni' or 'cheb' mode!")


                