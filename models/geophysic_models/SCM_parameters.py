"""DICE approx for Non-CO2 GHGs
"""

#: 2020 forcings of non-CO2 GHG (Wm-2)
FORCING_NONCO2_0 = 0.5 

#: 2100 forcings of non-CO2 GHG (Wm-2)                                  
FORCING_NONCO2_100 = 1.0                                     

try: 
    from ...parameters import T, STEP

except:
        T = 37
        STEP = 5
    
import numpy as np

F_EX  = np.ones(T*10) * FORCING_NONCO2_0               
"""Exogeneous forcing for other greenhouse gases (per period)

    :meta hide-value:
"""
for t in range(1,T*10):  
    F_EX [t] = FORCING_NONCO2_0 + min((FORCING_NONCO2_100 - FORCING_NONCO2_0), (FORCING_NONCO2_100 - FORCING_NONCO2_0)/T * t)


#: Carbon emissions from land 2015 (GtCO2 per year) 
EMISSION_LAND_0 = 2.6     

#: Decline rate of land emissions (per period)
DECLINE_RATE_EMISSION_LAND = .115       


E_EX = np.zeros(T*10)
"""Emission of land (per period)

    :meta hide-value:
"""
for t in range(T*10):
    E_EX[t] = EMISSION_LAND_0 * (1-DECLINE_RATE_EMISSION_LAND)**(t*STEP)
