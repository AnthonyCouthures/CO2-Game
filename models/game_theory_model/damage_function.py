import numpy as np
from models.geophysic_models import *

# def damage_polynome(coefficients : np.ndarray) -> callable :
#     """Function which create a damage function.

#     Parameters
#     ----------
#     coefficients : np.ndarray
#         Coefficient of the polynoial.

#     Returns
#     -------
#     callable
#         Damage function.
#     """
#     def damage(sum_action, **kwargs):
        
#         temp = kwargs.get('temp', None)

#         if temp is None :
#             smc : Simple_Climate_Model = kwargs.get('smc', None)
#             temp = smc.five_years_atmospheric_temp(sum_action)
#         return np.polyval(np.flip(coefficients),temp)
#     return damage

def damage_polynome(coefficients : np.ndarray) -> callable : 
    """Function which create a damage function.

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficient of the polynoial.

    Returns
    -------
    callable
        Damage function. 

        Parameters
        
        sum_action : float
            The total emissions of all players.

        Returns 

        float
            Damage for a given emission.

        Returns Other Parameters

        scm : Simple_Climate_Model
            Simple Climate model.
        temp : float
            Atmospheric temperature.
        GDP_max : float
            Maximum GDP attainable. 
        **kwargs : dict, optional
            Extra arguments to damage function.

    """

    def damage(sum_action, scm : Linear_Carbon_Model, **kwargs) -> float:
        """Damage function. 

        Parameters
        ----------
        sum_action : float
            The total emissions of all players.

        Returns
        -------
        float
            Damage for a given emission.

        Other Parameters
        ----------------
        scm : Simple_Climate_Model
            Simple Climate model.
        temp : float
            Atmospheric temperature.
        GDP_max : float
            Maximum GDP attainable. 
        **kwargs : dict, optional
            Extra arguments to damage function.

        """
        
        temp = kwargs.get('temp', None)
        gdp_max = kwargs.get('GDP_max', 100)

        if temp is None :
            temp = scm.five_years_atmospheric_temp(sum_action)
        d = np.polyval(np.flip(coefficients),temp)
        return gdp_max/100 * d 
    return damage




# def damage_function_economists(coefficients : np.ndarray) -> callable :
#     poly = np.polynomial.Polynomial(coefficients)
#     def damage(sum_action, smc : Simple_Climate_Model, temp = None):
#         if type(temp) == NoneType :
#             temp = smc.five_years_atmospheric_temp(sum_action)
#         return (sum_action * poly(temp))/(1 + sum_action * poly(temp))
    
#     return damage

# def DICE_damage(coefficients : np.ndarray, K = 1) -> callable :
#     def damage(sum_action : float, smc : Simple_Climate_Model, temp = None) :
#         if type(temp) == NoneType :
#             temp = smc.five_years_atmospheric_temp(sum_action)
#         poly = np.polynomial.Polynomial(coefficients)
#         return (poly(temp))/( K + poly(temp))
#     return damage

# def DICE_damage_v1(coefficients : np.ndarray, K = 1, power = 1) -> callable :
#     def damage(sum_action : float, smc : Simple_Climate_Model, temp = None) :
#         if type(temp) == NoneType :
#             temp = smc.five_years_atmospheric_temp(sum_action)
#         poly = np.polynomial.Polynomial(coefficients)
#         return (1)/( K + (poly(temp)**power))
#     return damage

# def damage_polynome(coefficients : np.ndarray) -> function :
#     def damage(sum_action, smc : Simple_Climate_Model):
#         temp = smc.five_years_atmospheric_temp(sum_action)
#         # print('temp :', temp)

#         # print('num_cycle :', smc.num_cycle)
#         damage_poly = np.polynomial.Polynomial(coefficients)
#         return damage_poly(temp)
#     return damage



# def damage_function_economists(coefficients : np.ndarray) -> function :
#     poly = np.polynomial.Polynomial(coefficients)
#     def damage(sum_action, smc : Simple_Climate_Model):
#         temp = smc.five_years_atmospheric_temp(sum_action)
#         return (sum_action * poly(temp))/(1 + sum_action * poly(temp))
    
#     return damage

# def DICE_damage(coefficients : np.array, K = 1) -> function :
#     def damage(sum_action : float, smc : Simple_Climate_Model) :
#         temp = smc.five_years_atmospheric_temp(sum_action)
#         poly = np.polynomial.Polynomial(coefficients)
#         return (poly(temp))/( K + poly(temp))
#     return damage

# def DICE_damage_v1(coefficients : np.array, K = 1, power = 1) -> function :
#     def damage(sum_action : float, smc : Simple_Climate_Model) :
#         temp = smc.five_years_atmospheric_temp(sum_action)
#         poly = np.polynomial.Polynomial(coefficients)
#         return (1)/( K + (poly(temp)**power))
#     return damage
