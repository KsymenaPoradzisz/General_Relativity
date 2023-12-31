import numpy as np
import numpy.typing as npt


class Data: 
    def __init__(self, size: list) -> None:
        self.grid = np.zeros(size).reshape( (-1) )
    

class BetaR(Data):
    # for the sake of numerical stability we choose \BetaR/R
    def dr(self) -> npt.NDArray:
        pass


class KRTheta(Data):
    # for the sake of numerical stability we choose KRTheta/R
    pass


class BetaU(Data):
    # for the sake of numerical stability we choose \BetaU/sqrt(1-u^2)
    pass


class Alpha(Data):
    # for the sake of numerical stability we choose \Alpha/R^2
    pass


class Eta(Data):
    # for the sake of numerical stability we choose \Eta/R
    pass

    