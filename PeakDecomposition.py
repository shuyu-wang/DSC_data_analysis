import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simps

# import modules published online
from lmfit.models import ExpressionModel


### Single Gaussian decomposition or multi-Gaussian decomposition of peak shape ###
# 1-d Gaussian model
def gaussian(x, amp, cen, wid):
    return amp*np.exp(-((x-cen)/wid)**2)
    #return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

# Root mean square error
def RMSE(target, prediction):
    
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))
    
    Rmse = np.sqrt(sum(squaredError) / len(squaredError))
    
    return Rmse


# Peak signal and Peak area temperature
# Single Gaussian model fitting
def PeakDecom(x, y):
    
    print("***Peak Fitting......")
    x = np.array(x)
    x = x.tolist()
    y = np.array(y)
    cen = min(x)
    Gmod = ExpressionModel("amp * exp(-(x-cen)**2 /(2*wid**2))/(sqrt(2*pi)*wid)")
    result = Gmod.fit(y, x=x, amp=1, cen=cen, wid=0.5)
    print(result.fit_report())
    
    plt.figure(dpi=600, figsize=(8,6))
    plt.plot(x, y, 'C2.', label='Net Signal')
    plt.plot(x, result.best_fit, 'C3-', label='Gaussian fitting')
    
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Temperature (℃)", size=22, labelpad=10)
    plt.ylabel("Heat capacity (KJ/mol/K)", size=22, labelpad=10)
    plt.tick_params(labelsize=16)
    plt.show()
    
    GauBestFit = result.best_fit 
    # RMSE
    Rmse = RMSE(y, GauBestFit)
    print("RMSE = ",Rmse)
    
    # Enthalpy change of Gaussian fitting
    print(" Enthalpy change of Gaussian fitting 1:", simps(GauBestFit, x) )


# 2-d Gaussian model
def func2(x, a1, a2, m1, m2, s1, s2):
	return a1*np.exp(-((x-m1)/s1)**2)+a2*np.exp(-((x-m2)/s2)**2)  

# Multiple Gaussian models fit peak signal
def MulPeakDecom(x, y):
    
    print("***Peak Fitting......")
    x = np.array(x)
    y = np.array(y)
    
    # set parammers
    AmpMin = 0.01
    AmpMax = 20000
    CenMin = min(x)
    CenMax = max(x)
    WidMin = 0.1
    WidMax = 100

    popt, pcov = curve_fit(func2, x, y, 
                           bounds=([AmpMin,AmpMin, CenMin,CenMin, WidMin,WidMin],
                                   [AmpMax,AmpMax, CenMax,CenMax, WidMax,WidMax]))
    #print(popt)
    #print(pcov)
    # Gaussian model parameters
    print("[[Variables]]")
    for i in range(2): # range(n) gaussian model number
        print("Model Variables", i, ":")
        print("  amp", i, ":", popt[i])
        print("  cen", i, ":", popt[i+2])
        print("  wid", i, ":", popt[i+4])
    
    # Plot peak fitting
    plt.figure(dpi=600, figsize=(8,6))
    plt.plot(x, y, "C2.", label='Net Signal')
    plt.plot(x, func2(x,*popt), 'C3-', label='Gaussian Fitting')
    plt.plot(x, gaussian(x, popt[0], popt[2], popt[4]), 'C4--', label= 'Gausssian 1')
    plt.plot(x, gaussian(x, popt[1], popt[3], popt[5]), 'C5--', label= 'Gausssian 2')
        
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Temperature (℃)", size=22, labelpad=10)
    plt.ylabel("Heat capacity (KJ/mol/K)", size=22, labelpad=10)
    plt.tick_params(labelsize=16)
    plt.show()
    
    # RMSE
    GauBestFit = gaussian(x, popt[0], popt[2], popt[4]) + gaussian(x, popt[1], popt[3], popt[5])
    Rmse = RMSE(y, GauBestFit)
    print("RMSE = ",Rmse)
    
    # Enthalpy change of Gaussian fitting
    print(" Enthalpy change of Gaussian fitting 1:", simps(gaussian(x, popt[0], popt[2], popt[4]), x) )
    print(" Enthalpy change of Gaussian fitting 2:", simps(gaussian(x, popt[1], popt[3], popt[5]), x) )
    


  
    
    
    
    