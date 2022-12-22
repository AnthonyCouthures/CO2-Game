import numpy as np
from .constants import C_1750, F_2XCO2

def radiative_forcing_function(atmospheric_carbon : float, equilibrium_atmospheric_carbon :float = C_1750,  forcing_2xco2 : float = F_2XCO2) -> float:
    """Function that calculate the radiative forcing of CO2 for a given quantity of carbon in atmosphere

    Parameters
    ----------
    atmospheric_carbon : float
        Quantity of CO2 in Gt Carbon in atmosphere in GtC
    equilibrium_atmospheric_carbon : float, optional
        The estimated carbon in atmosphere at equilibrium in GtC, by default C_1750
    forcing_2xco2 : float, optional
        Radiative forcing resulting from a doubling of atmospheric CO2, by default F_2XCO2

    Returns
    -------
    float
        Radiative forcing of the CO2 in W/m²
    """


    return forcing_2xco2 * np.log2(atmospheric_carbon/equilibrium_atmospheric_carbon) 


