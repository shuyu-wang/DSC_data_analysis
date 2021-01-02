import numpy as np
import numpy.linalg as la
from scipy.stats import skew, moment, norm
from scipy.special import erfcx, erfc

# import own module
import GradientDescent as GD
from LocationScaleProbability import LSPD
from SpecialFunctions import log_erfc


###### Standard Exponential modified Gaussian(ExpG) model ######
class Model:

    # standard parameterization of ExpG model
    def __init__(self, a = 1, optA = False):
        self.a = a  
        # Tracking whether or not a is optimized, boolean 
        self.optA = optA   


    # Check 
    def SetA(self, a):
        self.a = a

    def GetA(self):
        return self.a

    def SetOptA(self, optA):
        self.optA = optA

    def GetOptA(self):
        return self.optA

    # Define operators on ExpG distribution
    def __add__(self, other):
        return Model(self.a + other.a, self.optA)

    def __sub__(self, other):
        return Model(self.a - other.a, self.optA)

    def __mul__(self, scalar):
        return Model(self.a * scalar, self.optA)

    def __truediv__(self, other):
        return Model(self.a / other.a, self.optA)

    def norm(self):
        return la.norm(self.a)

    def print(self):
        print('a = ' + str(self.a))
        print('optA = ' + str(self.optA))

    def IsValid(self):
        return True if self.a > 0 else False

    def MakeValid(self, thres = 1e-6):
        self.a = max(self.a, thres)

    def Assign(self, other):
        self.a = other.a

    # Generate a standard normal distribution array + exponential distribution array = standard ExpG distributed
    def GenSamples(self, size = 1):
        return np.random.standard_normal(size) + np.random.exponential(scale = 1/self.a, size = size)

    # Negative log likelihood function. Standard Exponential Modified Gaussian Distribution
    def NegLogDen(self, x):
        a = self.a
        nld = -np.log(a/2) - a**2 / 2 + a*x - log_erfc( (a-x) / np.sqrt(2) )
        return nld

    def Density(self, x):
        # Likelihood function
        return np.exp(-self.NegLogDen(x))

    def IsConvexInA(self):
        return (True if self.a < 1 else False)

    def _get_d(self, x):
        a = self.a
        d = (a-x) / np.sqrt(2)
        return d

    def _get_de(self, x):
        d = self._get_d(x)
        e = 1 / erfcx(d)
        return d, e

    # First derivative of negative log density w.r.t. x
    def GradX(self, x):
        a = self.a
        d, e = self._get_de(x)
        return a - np.sqrt(2/np.pi) * e

    # Second derivative of standard negative log density w.r.t. x
    def GradX2(self, x):
        a = self.a
        d, e = self._get_de(x)
        return (2/np.pi * e**2 - 2/np.sqrt(np.pi) * e * d)

    # First derivative of standard negative log density w.r.t. a
    # a: alpha scalar
    def GradA(self, x):
        a = self.a
        d, e = self._get_de(x)
        return -(1/a + a) + x + np.sqrt(2/np.pi) * e

    # Second derivative of standard negative log density w.r.t. a
    def GradA2(self, x):
        a = self.a
        d, e = self._get_de(x)
        return (1/a**2 - 1) + 2/np.pi * e**2 - 2/np.sqrt(np.pi) * e * d

    # Gradient descent. Distribution parameter a
    def Gradient(self, x):
        return Model(np.sum(self.GradA(x)) if self.optA else 0)

    # On the value of the second derivative
    def Laplacian(self, x):
        return Model(np.sum(self.GradA2(x)) if self.optA else 0)

    def ScaledGradient(self, x, d = 1e-12):
        return Model(np.sum(self.GradA(x)) / (abs(np.sum(self.GradA2(x)) + d)) if self.optA else 0)

# The locationScaleFamily.py module was introduced. From the Gaussian 
# exponential model of the standard distribution to the exponential modified Gaussian model.
class ExpG( LSPD ):
    
    def __init__(self, a = 1, m = 0, s = 1, optA = False, optM = False, optS = False):
        self.m = m  # location parameter
        self.s = s  # scale parameter
        self.std = Model(a, optA)  # the standard distribution on which we are basing the LSPDamily

        # Boolean variables tracking which varialbes are optimized
        self.optM = optM
        self.optS = optS

    # getMS() comes from the LSPD module
    def GetAMS(self):
        return self.std.a, self.GetMS()

    def SetAMS(self, a, m, s):
        self.std.SetA(a)
        self.m = m
        self.s = s

    def SetOpt(self, optA, optM, optS):
        self.std.SetOptA(optA)
        self.optM = optM
        self.optS = optS

    def SetA(self, a):
        self.std.SetA(a)

    def SetOptA(self, optA):
        self.std.SetOptA(optA)

    def GradA(self, x):
        m, s = self.GetMS()
        return self.std.GradA((x-m)/s)

    def GradA2(self, x):
        m, s = self.GetMS()
        return self.std.GradA2((x-m)/s)


###### demo ######
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    E = ExpG()
    E.print()
    n = 1024
    x = E.GenSamples(n)
    '''
    import matplotlib.pyplot as plt 
    dom = np.linspace(-1, 1, n)
    plt.plot(dom, x)
    plt.show()
    '''
    E.SetAMS(.5, 0, 1)
    E.print()

    E.SetOptA(True)
    E.SetOptM(False)
    E.SetOptS(False)
    '''
    E.setOptA(True)
    E.setOptM(True)
    E.setOptS(True)
    '''
    E.print()
    E.Optimize(x)
    E.print()
