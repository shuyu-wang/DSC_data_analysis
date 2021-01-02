import numpy as np
import numpy.linalg as la
from copy import deepcopy

# import own module
import GaussModel as Gauss
import ExpGauss as ExpG
from LocationScaleProbability import LSPD


###### Standard Exponential modified Gaussian Mixture (ExpGMix) model ######
class Model():

    def __init__(self, a = 1, z = 0, optA = False, optZ = False):
        self.ExpG = ExpG.Model(a, optA)
        self.Gauss = Gauss.Model()
        # Indicator function
        self.z = z  
        # Tracking whether or not a is optimized, boolean 
        self.optZ = optZ 

    def print(self):
        self.ExpG.print()
        print('z = ' + str(self.z))
        print('optZ = ' + str(self.optZ))

    def GetZ(self):
        return self.z

    def SetZ(self, z):
        self.z = z

    def GetAZ(self):
        return self.a, self.z

    def SetAZ(self, a, z):
        self.ExpG.SetA(a)
        self.z = z

    def SetOpt(self, optA, optZ):
        self.ExpG.optA = optA
        self.optZ = optZ

    # Check
    def IsValid(self):
        if self.ExpG.IsValid() and self.Gauss.IsValid():
            return True
        else:
            return False

    def MakeValid(self, thres = 1e-6):
        self.ExpG.MakeValid()
        self.Gauss.MakeValid()
        return self

    # Assign the variables of another LSPDamily object to the current one
    def Assign(self, other):
        self.ExpG.Assign(other.ExpG)
        self.z = other.z

    # Define operators on standard ExpGaussMix objects
    def __add__(self, other):
        return Model(self.ExpG.GetA() + other.ExpG.GetA(), self.z + other.z, self.ExpG.GetOptA(), self.optZ)

    def __sub__(self, other):
        return Model(self.ExpG.GetA() - other.ExpG.GetA(), self.z - other.z, self.ExpG.GetOptA(), self.optZ)

    # Ideally would have scalar and elementwise multiplication
    def __mul__(self, scalar):
        return Model(self.ExpG.GetA() * scalar, self.z * scalar, self.ExpG.GetOptA(), self.optZ)

    def __truediv__(self, other):
        return Model(self.ExpG.GetA() / other.ExpG.GetA(), self.z / other.z, self.ExpG.GetOptA(), self.optZ)

    # Calculate the norm of the object
    def norm(self):
        return np.sqrt(self.ExpG.norm()**2 + la.norm(self.z)**2)

# Calculate the derivative of the negative log-likelihood function of the ExpGaussMix distribution, 
# which is convenient for gradient calculation to solve the minimum value. That is, the probability is the greatest.
    def NegLogDen(self, x):
        z = self.GetZ()
        return (1-z) * self.Gauss.NegLogDen(x) + z * self.ExpG.NegLogDen(x)
    
    def Density(self, x):
        return np.exp(-self.NegLogDen(x))

    def GradX(self, x):
        z = self.GetZ()
        return (1-z) * self.Gauss.GradX(x) + z * self.ExpG.GradX(x)

    def GradX2(self, x):
        z = self.GetZ()
        return (1-z) * self.Gauss.GradX2(x) + z * self.ExpG.GradX2(x)

    def GradA(self, x):
        z = self.GetZ()
        return z * self.ExpG.GradA(x)

    def GradA2(self, x):
        z = self.GetZ()
        return z * self.ExpG.GradA2(x)

    def Gradient(self, x):
        return Model(np.sum(self.GradA(x)) if self.ExpG.GetOptA() else 0, 0)

    def Laplacian(self, x):
        return Model(np.sum(self.GradA2(x)) if self.ExpG.GetOptA() else 0, 0)

    def ScaledGradient(self, x, d = 1e-12):
        return Model(np.sum(self.GradA(x)) / (abs(np.sum(self.GradA2(x)) + d)) if self.ExpG.GetOptA() else 0, 0)

    # Generate a standard ExpG mixture distribution
    def genSamples(self, size = 1):
        z = self.GetZ()
        ind = np.random.random(size) < z
        return (1-ind) * self.Gauss.GenSamples(size) + ind * self.ExpG.GenSamples(size)

    # Calculate the expected value of z, given mixture probabilities mix
    def ExpectedZ(self, x, mix):

        # compute responsibilities
        GauDen = self.Gauss.Density(x)
        ExpGDen = self.ExpG.Density(x)

        # Return (mix * ExpGDen) / ((1-mix) * GauDen + mix * ExpGDen)
        # for numerical stability
        ind = (mix*ExpGDen + (1-mix)*GauDen) == 0
        notInd = np.logical_not(ind)
        z = np.zeros(x.shape)
        z[ notInd ] = (mix * ExpGDen[notInd]) / (mix * ExpGDen[notInd] + (1-mix) * GauDen[notInd])
        z[ np.logical_and(ind, x>0) ] = 1
        z[ np.logical_and(ind, x<0) ] = 0
        return z

###### Exponential modified Gaussian mixture model ######
class ExpGaussMix( LSPD ):


    # could be a child of an abstract mixture model class
    def __init__(self, a = 1, m = 0, s = 1, z = 0, optA = False, optM = False, optS = False, optZ = False):

        self.std = Model(a, z, optA, optZ)
        self.m = m   # location parameter
        self.s = s   # scale parameter

        self.optM = optM
        self.optS = optS

    def GetAMSZ(self):
        a, z = self.std.GetAZ()
        return a, self.GetMS(), z

     # Set variables, makes sure that both distributions are updated
    def SetAMSZ(self, a, m, s, z):
        self.std.SetAZ(a, z)
        self.SetMS(m, s)

    def GetZ(self):
        return self.std.GetZ()

    def GetOptZ(self):
        return self.std.optZ

    def SetZ(self, z):
        return self.std.SetZ(z)

    # setting all optimization indicators
    def SetOpt(self, optA, optM, optS, optZ):
        self.std.SetOpt(optA, optZ)
        self.SetOptM(optM)
        self.SetOptS(optS)



    ####### Learning ######
    def CalculateMix(self):
        return np.mean(self.getZ())

    def ExpectedZ(self, x, mix):
        m, s = self.GetMS()
        return self.std.ExpectedZ( (x-m)/s, mix)   

    def ExpectationStep(self, x, mix):  
        self.SetZ(self.GetOptZ() * self.ExpectedZ(x, mix))  
        return self

    def MaximizationStep(self, x, mix, optMix, maxIter):

        # optimize continuous parameters of ExpGaussMix model
        super(ExpGaussMix, self).Optimize(x, maxIter = maxIter)

        # optimize mixture coefficient  
        if optMix:
            mix = np.mean(self.GetZ())  

        return mix

###### EM algorithm learning Exp correction Gaussian model mixture, peak signal intensity distribution ######
    def Optimize(self, x, mix = 1/2, optMix = True, maxIter = 8, minChange = 1e-6, maxMaxIter = 128):
        converged = False
        iter = 0
        while not converged:

            oldSelf = deepcopy(self)

            mix = self.MaximizationStep(x, mix, optMix, maxMaxIter)

            if np.any(self.GetOptZ()):
                self.ExpectationStep(x, mix)

            iter += 1
            if iter > maxIter or (oldSelf-self).norm() < minChange:
                converged = True
        '''
        print("location parameter：m=",self.m)
        print("scale parameter：s=",self.s)
        print("exponentially：a=",self.std.EMG.a) # Added by myself
        '''
        
        return self


###### Sample probability signal distribution demonstration ######
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    file = r'../combined_dsc1.csv' # BSA dsc data
    #file = r'../calfitter_dsc2.csv'# LinB dsc data, Calfitter
    #file = r'../lysozyme_dsc1.csv' # lysozyme dsc data

    
    df = pd.read_csv(file, header = None)
    df = df.T
    
    Q = df.iloc[0,1:]
    Q = Q.astype(float)
    Q = np.array(Q,dtype=np.float32)
    
    I = df.iloc[1:, 1:]
    #I = abs(I.astype(float))
    I = I.astype(float)
    I =np.array(I,dtype=np.float32)

    x = I
    
    # Lysozyme, normalization
    #x = -I
    #x = x * 17160 
    
    # BSA, normaliztion
    #x = I * 687.70764 

    mix=0.5
    ParVariable = ExpGaussMix(a=1,z=mix)
    dom=np.zeros(x.shape)
    GaussIsIn = np.ones(x.shape)
    for j in range(len(x)):
        ParVariable.SetOpt(True, True, True, True)
        ParVariable.Optimize(x[j],mix)
        ParChange = ParVariable.GetZ()
        
        for k in range(len(ParChange)):
            #if ParChange[k] > mix+0.02 :  # Lysozyme dsc data
            if ParChange[k] > mix+0.007:
                #ParChange[k] = 1
                dom[j,k] = 1
        fig = plt.figure(figsize=(6,4))
        ax1 = fig.add_subplot(111)
        plot1 = ax1.plot(Q, x[j], c='C1', label='Measuring signal')
        ax2 = ax1.twinx()
        plot2 = ax2.plot(Q, dom[j], c='C0', label='Distribution probability')
        lines = plot1 + plot2
        ax1.legend(lines, [l.get_label() for l in lines])
        plt.legend(loc='best')
        ax1.set_xlabel("Temperature", size=18)
        ax1.set_ylabel("Heat capacity", size=18)
        ax2.set_ylabel("Distribution probability", size = 18)
        plt.title("scan=%s"%(j))
        plt.show()

    '''
    print('Inferred mixture probability:' + str(np.mean(ExpGaussMix.getZ())))
    print('True mixture probability:' + str(mix))
    ExpGaussMix.print()
    '''
      








