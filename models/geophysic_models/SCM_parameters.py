"""DICE approx for Non-CO2 GHGs
"""
import pandas as pd
import pathlib
pathlib.Path(__file__).parent.resolve()
import sys

sys.path.insert(0, pathlib.Path(__file__).resolve().as_posix())
#: 2020 forcings of non-CO2 GHG (Wm-2)
FORCING_NONCO2_0 = 0.5 

#: 2100 forcings of non-CO2 GHG (Wm-2)                                  
FORCING_NONCO2_100 = 1.0                                     

tmax = 67
step = 5
import numpy as np

F_EX  = np.ones(tmax*10) * FORCING_NONCO2_0               
"""Exogeneous forcing for other greenhouse gases (per period)

    :meta hide-value:
"""
for t in range(1,tmax*10):  
    F_EX [t] = FORCING_NONCO2_0 + min((FORCING_NONCO2_100 - FORCING_NONCO2_0), (FORCING_NONCO2_100 - FORCING_NONCO2_0)/tmax * t)

# From REMIND IAM  
F_EX = np.array([0.310,0.455,0.514,0.580,0.586,0.592,0.562,0.533,0.496,0.462,0.443,0.425,0.411,0.397,0.382,0.367,0.352,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337,0.337])

#: Carbon emissions from land 2015 (GtCO2 per year) 
EMISSION_LAND_0 = 2.6     

#: Decline rate of land emissions (per period)
DECLINE_RATE_EMISSION_LAND = .115       


E_EX = np.zeros(tmax*10)
"""Emission of land (per period)

    :meta hide-value:
"""
for t in range(tmax*10):
    E_EX[t] = EMISSION_LAND_0 * (1-DECLINE_RATE_EMISSION_LAND)**(t*step)
