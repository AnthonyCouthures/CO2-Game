import numpy as np
from models.geophysic_models import *
from collections.abc import Callable

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

    def damage(temp, **kwargs) -> float:
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
        **kwargs : dict, optional
            Extra arguments to damage function.

        """
        

        if temp is None :
            sum_action = kwargs.get('sum_action', None)
            scm : Simple_Climate_Model = kwargs.get('scm', None)
            if type(sum_action) is not np.float64 :
                temp = scm.evaluate_trajectory(sum_action, **kwargs)
            else :
                temp = scm.evaluate_trajectory(np.array([sum_action]))[-1][0]
        return np.polyval(np.flip(coefficients),temp)
    return damage


def damage_polynome(temp, coefficients : np.ndarray, **kwargs) -> Callable : 

    

    if temp is None :
        sum_action : np.ndarray = kwargs.get('sum_action', None)
        scm : Simple_Climate_Model = kwargs.get('scm', None)
        if type(sum_action) is not np.float64 :
            temp = scm.evaluate_trajectory(sum_action, **kwargs)
        else :
            temp = scm.evaluate_trajectory(np.array([sum_action]))[-1][0]
    return np.polyval(np.flip(coefficients),temp)

def damage_exponential(coefficients : np.ndarray) : 
    def damage(temp, **kwargs) -> float:
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
        **kwargs : dict, optional
            Extra arguments to damage function.

        """
        

        if temp is None :
            sum_action = kwargs.get('sum_action', None)
            scm : Simple_Climate_Model = kwargs.get('scm', None)
            if type(sum_action) is not np.float64 :
                temp = scm.evaluate_trajectory(sum_action, **kwargs)
            else :
                temp = scm.evaluate_trajectory(np.array([sum_action]))[-1][0]
        return np.polyval(np.flip(coefficients),temp)
    return damage

def damage_exponential(temp, coefficients : np.ndarray, **kwargs) : 


    if temp is None :
        sum_action = kwargs.get('sum_action', None)
        scm : Simple_Climate_Model = kwargs.get('scm', None)
        if type(sum_action) is not np.float64 :
            temp = scm.evaluate_trajectory(sum_action, **kwargs)
        else :
            temp = scm.evaluate_trajectory(np.array([sum_action]))[-1][0]
    return coefficients[0] * np.exp(temp * coefficients[1]) + coefficients[2]


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
