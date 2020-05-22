from enum import Enum
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.neighbors import KNeighborsRegressor as KNR

class Algorithms(Enum):
    RandomForestRegressor = RFR()
    MLPRegressor = MLPR()
    KNeighborsRegressor = KNR(n_neighbors=5, weights='distance')

    def __str__(self):
        return self.name