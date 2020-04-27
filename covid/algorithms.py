from enum import Enum
from covid.algorithm import Algorithm
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neural_network import MLPRegressor as MLPR

class Algorithms(Enum):
    RandomForestRegressor = RFR()
    MLPRegressor = MLPR()

    def __str__(self):
        return self.name