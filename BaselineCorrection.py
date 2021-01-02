import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from copy import deepcopy

# import own module
from ExpGaussMix import ExpGaussMix
from PeakDecomposition import PeakDecom
from PolyBaseline import PolyFit



###### Baseline correction ######
class BaselineCorrection():
    
    def __int__(self, Tm, Cp):
        
        self.Tm = Tm
        self.Cp = Cp
    
# Find the index value at the center of the peak area
    def PeakArea(self, Tm, Cp, dom=1):

        # EM Algorithm for optimization of ExpGauss Mixture.  
        # Probability classification of sample signals
        x = Cp
        mix=0.5
        ParVariable = ExpGaussMix(a=0.7, z=mix)
        dom = np.zeros(x.shape)  
        GaussIsIn = np.ones(x.shape)
        for j in range(len(x)):
            ParVariable.SetOpt(True, True, True, True)
            ParVariable.Optimize(x[j], mix)
            ParChange = ParVariable.GetZ()

            # Probability classification of sample signals
            for k in range(len(ParChange)):
                #if ParChange[k] > mix+0.007: # for BSA, LinB DSC data
                #if ParChange[k] > mix+0.02: # for Lysozyme DSC data
                if ParChange[k] > mix+0.005: # for BSA, LinB DSC data
                    dom[j,k] = 1
                 
        # Coarse positioning peak area. z=1 is Peak area, z=0 is Background signal
        Peak1 = dom.tolist()
        Peak = []
        for i in range(0, dom.shape[0]):
            PeakIdex = [i for i,num in enumerate(Peak1[i]) if num==1]
            Peak.append([])
            Peak[i] = PeakIdex
        
        # Center position of peak signal area
        Peak0 = []
        for i in range(len(Peak)):
            Peak0.append([])
            sum = 0
            for j in Peak[i]:
                sum = sum + j
            sum = sum/(len(Peak[i])+0.00001)
            Peak0[i] = sum
             
        return Peak0,dom
    

# Find the left edge of the peak,the right edge of the peak.
# The maximum value of BePeak is the left peak index 
    def BePeak(self, Tm, Cp, fit_num=4):
        
        x = Tm
        y = Cp
        PeakBas1 = np.zeros (y.shape)
        for i in range (0,y.shape[0]):
            fit_y = y[i]
            PeakBas1[i] = PolyFit(x, fit_y, fit_num)
            
        Peak, dom = self.PeakArea(x, y)
        Peak = np.array(Peak)
        BePeak = np.zeros(y.shape)
        for i in range(0, y.shape[0]):
            k = Peak[i]
            k = int(k)
            for j in range(0,k):
                a = PeakBas1[i,k-j] - y[i,k-j]
                if a == 0:   
                    BePeak[i,j] = k-j
                    
        AfPeak = np.zeros(y.shape)
        for i in range(0, y.shape[0]):               
            k = Peak[i]
            k = int(k)
            for j in range(k, y.shape[1]):
                b = PeakBas1[i,j] - y[i,j]
                if b == 0:
                    AfPeak[i,j] = j
                    
        return BePeak,AfPeak,dom
    
    
def BaseCorrect(Tm, Cp, fit_num=4):
    
    print("------Creating DSC Object------")
    P = BaselineCorrection()
    x = Tm
    y = Cp
    
    # Polynomial fitting the baseline
    print("*Polynomial fitting produces baseline")
    PeakBas1 = np.zeros(y.shape)
    for i in range(0, y.shape[0]):
        fit_y = y[i]
        PeakBas1[i] = PolyFit(x, fit_y, fit_num)
    
    print("**Baseline correction")
    # polyfit the baseline coorection. Corrected peak area range
    BePeak, AfPeak, dom = P.BePeak(x, y)
    PeakBas = np.zeros(y.shape)
    
    RawBeBas = deepcopy(y)
    RawAfBas = deepcopy(y)
    
    for i in range (0, y.shape[0]):  
        # Peak left margin
        BePeak1 = BePeak[i]
        mB = max(BePeak1)
        mB = int(mB)
        
        # Peak right margin
        AfPeak1 = []
        for j in range(AfPeak.shape[1]):
            if AfPeak[i,j] != 0:
                AfPeak1.append(AfPeak[i,j])        
        mA = min(AfPeak1)
        mA = int(mA)
        
        # Baseline correction
        for j in range (mB, mA):
            PeakBas[i,j] = PeakBas1[i,j]
            RawBeBas[i,j] = 0
            RawAfBas[i,j] = 0
        for k in range(0, mB):
            RawBeBas[i,k] = y[i,k]
            RawAfBas[i,k] = 0   
        for l in range(mA,y.shape[1]):
            RawAfBas[i,l] = y[i,l]
            RawBeBas[i,l] = 0
                      
    # baseline of the peak area   
    RawBas = RawBeBas + RawAfBas
    EMGMBas = PeakBas + RawBas
    # Net Signal
    ResiData = y - EMGMBas
    
    # Peak signal and Peak area temperature
    Peak = []
    TemPeak = []
    for i in range(0, ResiData.shape[0]):
        Peak.append([])
        TemPeak.append([])
        for j in range(0,ResiData.shape[1]):
            if ResiData[i,j] != 0:
                Peak[i].append(ResiData[i,j]) 
                TemPeak[i].append(x[j]) 
                
    # Number of peaks
    NumPeak = np.zeros(shape = (ResiData.shape[0],1))
    for k in range(0, ResiData.shape[0]):
        Peak1 = np.array(Peak[k])
        num = 0
        for l in range(1, Peak1.shape[0]-1):
            if Peak1[l-1] < Peak1[l]  and  Peak1[l] > Peak1[l+1]: 
                num +=1
                NumPeak[k] = num 
                
    # return Net signal, Baseline correction, Probability, Peak signal, Peak Temperature area   
    return ResiData, EMGMBas, dom, Peak, TemPeak 








