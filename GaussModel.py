import numpy as np

# import own module
from LocationScaleProbability import LSPD


###### Standard Gaussian model ######
class Model():

    # Negative log likelihood function
    def NegLogDen(self, x):
        return x**2/2 + np.log(2*np.pi) / 2

    # Differentiation of the negative log likelihood function
    def GradX(self, x):
        return x

    # The second derivative of the negative log likelihood function
    def GradX2(self, x):
        return np.ones(x.shape)

    def Gradient(self, x):
        return self

    def Laplacian(self, x):
        return self

    def ScaledGradient(self, x, d = 1e-12):
        return self

    def print(self):
        return None

    def IsValid(self):
        return True

    def MakeValid(self):
        return self

    def Assign(self, other):
        return None

    def GenSamples(self, size = 1):
        return np.random.standard_normal(size)

    # Define operators on standard normal distribution
    def __add__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __mul__(self, scalar):
        return self

    def __truediv__(self, other):
        return self

    def norm(self):
        return 0

    # Functions regarding the standard normal distribution
    def Density(self, x):
        # Standard normal distribution probability density function
        return 1/(np.sqrt(2*np.pi)) * np.exp( - x**2 / 2 )

    def DenGradX(self, x):
        # First-order derivative of likelihood function
        return -x * self.Density(x)

    def DenGradX2(self, x):
        # Second-order derivative of the likelihood function
        return - self.Density(x) + (-x * self.DenGradX(x))

###### Gaussian Distribution ######
class Gauss( LSPD ):

    def __init__(self, m = 0, s = 1, optM = False, optS = False):
        self.std = Model()
        self.m = m
        self.s = s

        self.optM = optM
        self.optS = optS




       