import numpy as np
from copy import deepcopy


### Iterative polynomial fitting generate baseline ###
def PolyFit(fit_x, fit_y, fit_num = 5,pt = 1):
    
    fit_x = np.array(fit_x)
    fit_y = np.array(fit_y)
    fit_num = fit_num 
    f1 = np.polyfit(fit_x, fit_y, fit_num)  
            
    # Use np.poly1d() function to solve the fitted curve
    b0 = np.poly1d(f1) # b0 is the power of polynomial fitting
    b1 = list( map(b0, fit_x) )
    b = np.array(b1)
            
    pt = 1
    y0 = deepcopy(fit_y)
    bk = deepcopy(fit_y)
    while pt > 0.0001:
        for i in range(0,len(fit_y)-1):
            if y0[i] > b[i]:
                y0[i] = b[i]
        z = y0 -bk
        z0 = 0
        bn = 0
        for i in range(0,len(fit_y)-1):
            z0 = z0 + z[i]**2
            bn = bn + bk[i]**2
        pt = np.sqrt(z0)/np.sqrt(bn)
        f1 = np.polyfit(fit_x, y0,fit_num)
        b0 = np.poly1d(f1)
        b1 = list(map(b0, fit_x))
        b = np.array(b1)
        bk = deepcopy(y0)
    
    return y0