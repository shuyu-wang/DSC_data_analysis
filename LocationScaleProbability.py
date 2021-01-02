import numpy as np
import numpy.linalg as la

# import own module
import GradientDescent as GD


###### Standear location scale probability distribution ######
class LSPD:
    
    # Subclasses for density and nll for LSFD and GM
    def __init__(self, std, m = 0, s = 1, optM = False, optS = False):
        self.std = std  # standard distribution 
        self.m = m  # location parameter
        self.s = s  # scale parameter

        # Boolean variables tracking which varialbes are optimized
        # Optimized logo,, boolen variables
        self.optM = optM
        self.optS = optS

    def print(self):
        print('m = ' + str(self.m))
        print('optM =' + str(self.optM))
        print('s = ' + str(self.s))
        print('optS =' + str(self.optS))
        self.std.print()

    # checks if the distribution is valid
    def IsValid(self):
        if np.all(self.s > 0) and self.std.IsValid():
            return True
        else:
            return False

    def MakeValid(self, thres = 1e-6):
        self.std.MakeValid()
        self.s = np.maximum(thres, self.s) # comparison of parameters
        return self

    # Assign the variables of another LSPDamily object to the current one
    def Assign(self, other):
        self.std.Assign(other.std)
        self.m = other.m
        self.s = other.s

    # Define operators on location scale family
    def __add__(self, other):
        return LSPD(self.std + other.std, self.m + other.m, self.s + other.s, self.optM, self.optS)

    def __sub__(self, other):
        return LSPD(self.std - other.std, self.m - other.m, self.s - other.s, self.optM, self.optS)

    # Ideally would have scalar and elementwise multiplication
    def __mul__(self, scalar):
        return LSPD(self.std * scalar, self.m * scalar, self.s * scalar, self.optM, self.optS)

    def __truediv__(self, other):
        return LSPD(self.std / other.std, self.m / other.m, self.s / other.s, self.optM, self.optS)

    def norm(self):
        return np.sqrt(self.std.norm()**2 + la.norm(self.m)**2 + la.norm(self.s)**2)

    def GetMS(self):
        return self.m, self.s

    def SetMS(self, m, s):
        self.m = m
        self.s = s

    def SetM(self, m):
        self.m = m

    def GetM(self):
        return self.m

    def SetS(self, s):
        self.s = s

    def GetS(self):
        return self.s

    def SetOptM(self, optM):
        self.optM = optM

    def SetOptS(self, optS):
        self.optS = optS

    def SetOpt(self, optM, optS):
        self.optM = optM
        self.optS = optS

    # Generate location scale distribution sample
    def GenSamples(self, size = 1):
        if size > 1:
            return self.s * self.std.GenSamples(size) + self.m  
        elif size == 1:
            return self.s * self.std.GenSamples(self.m.shape) + self.m

    # Negative logarithmic density function
    def NegLogDen(self, x):
        m, s = self.GetMS()
        return self.std.NegLogDen((x-m)/s) + np.log(s)

    # The first and second order derivation of probability density function
    def GradM(self, x):
        m, s = self.GetMS()
        return -1/s * self.std.GradX((x-m)/s)

    def GradM2(self, x):
        m, s = self.GetMS()
        return 1/s**2 * self.std.GradX2((x-m)/s)

    def GradS(self, x):
        m, s = self.GetMS()
        xm = (x-m)/s
        return self.std.GradX(xm) * -xm/s + 1/s

    def GradS2(self, x):
        m, s = self.GetMS()
        xm = (x-m)/s
        return (self.std.GradX2(xm) * (xm/s)**2 + self.std.GradX(xm) * 2*xm/s**2) - 1/s**2

    # The first and second order derivation of probability density function
    def GradX(self, x):
        m, s = self.GetMS()
        return 1/s * self.std.GradX((x-m)/s)

    def GradX2(self, x):
        m, s = self.GetMS()
        return 1/s**2 * self.std.GradX2((x-m)/s) 


    # Use the pipeline function to move the sum
    def Gradient(self, x):
        gradM = np.sum(self.GradM(x)) if self.optM else 0   
        gradS = np.sum(self.GradS(x)) if self.optS else 0
        return LSPD(self.std.Gradient(x), gradM, gradS, self.optM, self.optS)

    def Laplacian(self, x):
        gradM2 = np.sum(self.GradM2(x)) if self.optM else 0
        gradS2 = np.sum(self.GradS2(x)) if self.optS else 0
        return LSPD(self.std.Laplacian(x), gradM2, gradS2, self.optM, self.optS)

    def ScaledGradient(self, x, d = 1e-12):
        gradM = np.sum(self.GradM(x)) / np.sum(self.GradM2(x)) if self.optM else 0
        gradS = np.sum(self.GradS(x)) / (abs(np.sum(self.GradS2(x))) + d) if self.optS else 0
        # in log domain, have to scale gradS by exp(log(s)) = s
        return LSPD(self.std.ScaledGradient(x), gradM, gradS, self.optM, self.optS)


    # Negative Log Likelihood 
    def NegLogLike(self, x):
        return np.sum(self.NegLogDen(x))


    # Parameter Estimation
    # Optimize parameters. Optimize parameters together according to data x
    def Optimize(self, x, maxIter = 32, plot = False):
        
        # defineOptimizationParameters() method from GD.py
        params = GD.DefineOptimizationParameters(maxIter = maxIter, minDecrease = 1e-5)
        obj = lambda E : E.NegLogLike(x)
        grad = lambda E : E.ScaledGradient(x)
        updateVariables = lambda E, dE, s : E - (dE * s)
        projection = lambda E : E.MakeValid()
        E, normArr, stepArr = GD.GradientDescent(self, obj, grad, projection, updateVariables, params)
        self.Assign(E)
        if plot:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.plot(normArr)
            plt.subplot(122)
            plt.plot(stepArr)
            plt.show()
        return self

    
    # Density 
    def Density(self, x):
        return np.exp(-self.NegLogDen(x))

    # Compute gradient of density given gradients of negative log-density
    def DenGrad(self, den, nllGrad):
        return den * -nllGrad

    def DenGrad2(self, den, nllGrad, nllGrad2):
        return den * (nllGrad**2 - nllGrad2)

    def DenGradX(self, x):
        return self.DenGrad(self.Density(x), self.GradX(x))

    def DenGradX2(self, x):
        return self.DenGrad2(self.Density(x), self.GradX(x), self.GradX2(x))

    def DenGradM(self, x):
        return self.DenGrad(self.Density(x), self.GradM(x))
        # return self.density(x) * -self.gradM(x)

    def DenGradM2(self, x):
        return self.DenGrad2(self.Density(x), self.GradM(x), self.GradM2(x))

    def DenGradS(self, x):
        return self.DenGrad(self.Density(x), self.GradS(x))

    def DenGradS2(self, x):
        return self.DenGrad2(self.Density(x), self.GradS(x), self.GradS2(x))


