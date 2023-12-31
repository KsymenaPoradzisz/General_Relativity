import numpy as np


# storage for derivative matrices
class Derivative:
    _matrix = None  # static variable to be easily accessible without redundant copying

    def __init__(self, size: list) -> None:
        self._matrix = np.zeros(size)


class DR(Derivative):
    def __init__(self, size: list) -> None:
        super().__init__(size)



class DTheta(Derivative):
    def __init__(self, size: list) -> None:
        super().__init__(size)  
         