import numpy as np
from scipy.special import erfcx, erfc


# Realization of Numerical Stability of Logarithmic Complementary Error Function
def log_erfc( x ):
    fx = np.zeros(x.shape)
    ind = (x > 8)
    fx[ind] = np.log(erfcx(x[ind])) - x[ind]**2
    ind = (x <= 8)
    fx[ind] = np.log(erfc(x[ind]))

    return fx
