from enum import Enum
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.linear_model import Ridge as RR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso as LR


class Algorithms(Enum):
    RandomForestRegressor = RFR()
    MLPRegressor = MLPR()
    KNeighborsRegressor = KNR()
    Ridge = RR()
    Lasso = LR()

    def __str__(self):
        return self.name
