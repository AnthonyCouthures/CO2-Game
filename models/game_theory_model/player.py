"""A player class.

Here we create a player class where wa concatenate characteristics of a player:

- Action set
- Utility function
- etc..

"""
from collections.abc import Callable

import numpy as np
from scipy.optimize import Bounds, minimize, NonlinearConstraint
from ..geophysic_models import *
from copy import deepcopy
from parameters import *
from .jacobian import * 
from functools import partial
from typing import Union
import time


class player_class:
    """Class of a player.
    """

    def __init__(self, name : str,
                action_set : np.ndarray,
                benefit_function : Callable, 
                GDP_initial : float,
                damage_function : Callable,
                impact_factor_of_temperature: float,
                scm : Simple_Climate_Model = SCM,
                alpha : float = 1,
                increase_co2 = INCREASE_CO2_RATIO,
                percentage_green = 0.0,
                damage_in_percentage = True,
                discount = 1) -> None:
                

        self.name : str = name
        "Name of the player"

        self.action_set_initial : np.ndarray = action_set
        "Initial action set of the player (e_min, e_max)"

        self.GDP_initial : float = GDP_initial
        "Initial GDP of the player, i.e the maximum a player can attain with his benefit function."

        self.percentage_green = percentage_green
        "Percentage of the benefit being decarbonated."
        
        self.benefit_shape : Callable = benefit_function
        "Benefit function shape of the player, it takes as argument (GDP, e_max, percentage_green)."

        self.GDP_max = deepcopy(GDP_initial)
        "Current GDP of the player, i.e the maximum a player can attain with his benefit function."

        self.action_set = deepcopy(action_set)   # deepcopy for create a new memory location
        "Current action set of the player (e_min, e_max)"

        self.T : int = action_set.shape[0]
        "Number of time periods"

        self.scm = deepcopy(scm)
        "Simple Climate Model of the player"


        self.damage_function = damage_function
        "Damage function, it takes as argument emissions."

        self.delta = impact_factor_of_temperature
        "Temperature multiplier the player"

        self.alpha = alpha
        "Power of the damage function"

        self.increase_co2 = increase_co2
        "Ration of player tend to increase his co2 emissions over a game if it polluted at maximum."

        self.damage_in_percentage : bool = damage_in_percentage
        "Specify if the damage is as percentage og GDP."

        self.discount = discount


    def reset_player(self):
            """Function which reset the player to the inital state for a given time period.

            Parameters
            ----------
            t : int
                The time period to reset the player for.
            """

            # deepcopy for reset due to pointer 
            self.action_set = deepcopy(self.action_set_initial)
            self.GDP_max = deepcopy(self.GDP_initial)


    def reset_scm(self) -> None:
        """Function which reset the SCM of the player.
        """
        self.scm.reset()


    def reset(self):
        """Function which reset the player and the SCM.
        """
        self.reset_player()
        self.reset_scm()

    def benefit(self, actions : np.ndarray, **kwargs):

        GDP_max = kwargs.get('GDP_max', self.GDP_max) # non t dependant yet
        e_max = kwargs.get('e_max', self.action_set[:,1]) #t dependant
        percentage_green = kwargs.get('percentage_green', self.percentage_green) # non t dependant yet
        t = kwargs.get('t', None)
        if t is None:
            t0 = kwargs.get('t0', 0)
            return np.array([self.benefit_shape(GDP_max, e_max[t0+t], percentage_green)(actions[t]) for t in range(len(actions))])
        else : 
            return self.benefit_shape(GDP_max, e_max[t], percentage_green)(actions)
        
    def damages(self, emissions, **kwargs):
        temperature_target = kwargs.pop('temperature_target', None)
        final_multiplier = kwargs.pop('final_multiplier',  FINAL_MULTIPLIER)

        atmospheric_temperatures = kwargs.get('temp', self.scm.evaluate_trajectory(emissions, **kwargs)[-1])  # C'est important car sinon on a le joueur qui evalue la trajectoire avec un SCM déjà modifier
        if self.damage_in_percentage:
            GDP_max = kwargs.pop('GDP_max',self.GDP_max)
            damages = self.delta  * GDP_max/100 * self.damage_function(atmospheric_temperatures)**self.alpha
        else:
            damages = self.delta  * self.damage_function(atmospheric_temperatures) **self.alpha

        if temperature_target is not None:
            damages = damages + final_multiplier * (atmospheric_temperatures[-1] - temperature_target)

        return damages


    def utility_one_shot(self, action : np.ndarray, sum_other_actions : np.ndarray, t : int, **kwargs) -> float :# temp : float = None, scm = False) -> float:
        """Utility of the player for a given time period.

        Parameters
        ----------
        action : float
            Emission of the player.
        sum_other_actions : float
            Emissions of other players.

        Other Parameters
        ----------------
        GDP_max : float, optional
            Maximum GDP attainable of the player. Often it will be remplace with self.GDP_max but in some case it's usefull to have it as a parameter.
        temp : float, optional
            Current temperature. If not provided, the player will generate a temperature with the given actions with the SCM. If given, it speed up the calculs, by default None
        scm :  optional
            SCM used by the player. If not provided, the player will generate a temperature with the given actions with this specific SCM.

        Returns
        -------
        float
            Utility of the player.
        """

        scm = kwargs.pop('scm', self.scm)
        temp = kwargs.pop('temp', scm.evaluate_trajectory(action + sum_other_actions, **kwargs)[-1])

        utility = self.benefit(action, t = t, **kwargs) -  self.damages(action + sum_other_actions, temp = temp, **kwargs)

        return utility


    def utility_over_t(self, emissions, others_emissions, **kwargs):
        if self.discount !=1:
            discount = self.discount**np.arange(len(emissions))
        else:
            discount = 1
        benefit = self.benefit(emissions, **kwargs)
        damages = self.damages(emissions + others_emissions, **kwargs)
        utility = discount * (benefit - damages)

        return utility


    def utility_sum_over_t(self, emissions, others_emissions, **kwargs):
        if self.discount !=1:
            discount = self.discount**np.arange(len(emissions))
        else:
            discount = 1
        benefit = self.benefit(emissions, **kwargs)
        damages = self.damages(emissions + others_emissions, **kwargs)
        utility = np.sum(discount * (benefit - damages))

        return utility



    def best_response_over_t(self, others_emissions, **kwargs):
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', len(others_emissions)) + t0
        def response(x):
            return -self.utility_sum_over_t(x, others_emissions, **kwargs)
        jac= self.jacobian_utility_sum_over_t(others_emissions, **kwargs)
        action_set = self.action_set[t0:tmax,:]
        # print(t0)
        # print(tmax)
        bounds = Bounds(lb = action_set[:,0], ub = action_set[:,1], keep_feasible=True) 
        # constraint = NonlinearConstraint(response, -1e9, 0, jac=jac)
        # print(kwargs['x0'])
        x0 = kwargs.get('x0', action_set[:,1])
        # print('x0', x0.shape, x0)
        # print('bounds', bounds) 




        # start_time = time.time()
        # res = minimize(response, x0 = x0, bounds=bounds,  method='L-BFGS-B', tol=1e-6, options={'maxiter': 10000, 'disp': 0}) #, constraints=constraint)#jac=jac,, constraints=constraint)
        # end_time = time.time()
        # print('L-BFGS-B:', end_time - start_time)
        res = minimize(response, x0 = x0, bounds=bounds,  method='SLSQP', tol=1e-6, options={'maxiter': 10000, 'disp': 0}) #, constraints=constraint)#jac=jac,, constraints=constraint)
        # start_time1 = time.time()
        # res = minimize(response, x0 = x0, bounds=bounds,  method='SLSQP', tol=1e-6, options={'maxiter': 10000, 'disp': 0}) #, constraints=constraint)#jac=jac,, constraints=constraint)
        # end_time1 = time.time()
        # print('SLSQP:', end_time1 - start_time1)
   
        # res = minimize(response, x0 = x0, bounds=bounds, constraints=constraint, jac=jac, options={'maxiter': 1000, 'disp': 0})
        # print('Without JAC', res)

        if not res.success:
            print(self.name)
            print(bounds)
            print(self.alpha)
            print(res)
        return res.x




    def best_response_one_shot(self, sum_other_actions : np.ndarray, t : int) -> float:
        """Best Response function of the player for a given sum of other players emissions.

        Parameters
        ----------
        sum_other_actions : float
            The sum of the others players emissions. 

        Returns
        -------
        float
            The best response of the player for a sum of other players emissions.

        Notes
        -----
        
        We use minimise scalar with the following parameters 
        (response, bounds=self.action_set, method='bounded', options={'xatol': 1e-05, 'maxiter': 1000, 'disp': 0}),
        where response is equal to minus the utility of the player.
        """

        def response(x):
            return -self.utility_one_shot(x, sum_other_actions, t)
        bounds = Bounds(lb=[self.action_set[t,0]], ub= [self.action_set[t,1]], keep_feasible=True)
        res = minimize(response, x0 = np.array([self.action_set[t,1]]),  method='SLSQP',  tol=1e-20,  bounds=bounds, options={ 'maxiter': 1000, 'disp': 0}) #,constraints =constraint)
        return res.x

    def update_scm(self, sum_actions : float, exogeneous_emission : float, exogeneous_radiative_forcing) -> None:
        self.scm.five_years_cycle_deep(sum_actions, exogeneous_emission, exogeneous_radiative_forcing)

    def jacobian_utility_sum_over_t(self, others_emissions, **kwargs):
        if self.discount !=1:
            discount = self.discount**np.arange(len(others_emissions))
        else:
            discount = 1
        t_periode = len(others_emissions)
        CC = self.scm.carbon_model
        TD = self.scm.temperature_model
        def jacobian(x):
            carbon_AT, forcing, temperature_AT = self.scm.evaluate_trajectory(x + others_emissions, **kwargs)
            jac_carbon_AT = jacobian_linear_model(CC.Ac, CC.bc, CC.dc, t_periode)
            jac_forcing = jacobian_forcing(carbon_AT)
            jac_temperature_AT = jacobian_linear_model(TD.At, TD.bt, TD.dt, t_periode)
            jac_damage = jacobian_damage_function(temperature_AT, self.damages)

            jac_benefit = np.array([derivative(partial(self.benefit, t=t),x[t], order=5, dx=1e-6) for t in range(t_periode)])
            return -discount * (jac_benefit - self.delta * jac_damage @ jac_temperature_AT @ jac_forcing @ jac_carbon_AT)
        return jacobian

    def update_player(self, previous_emission : np.ndarray, t : int) -> None:
        """Function which update the player from a previous state for a given previous emission.

        Notes
        -----

        Main place for improvement.

        Parameters
        ----------
        previous_emission : np.ndarray
            Emissions of the players in the previous iteration of the game. 
        t : int
            The time period to update the action set for.
        """
        if t >= self.T:
            raise ValueError("Invalid time period.")

        # We calculate the ratio of previous emission compare to what the player could have polluted.
        ratio = previous_emission[t] / self.action_set[t,1]

        # The action set of the player update
        self.action_set[1,t:] = self.action_set[t,1] * (1 + ratio * self.increase_co2)
        self.GDP_max = self.GDP_max * (1 + ratio*0.02)
    


class Invest_Player:
    """Class of a player that invests in order to reduce emissions.

    Parameters
    ----------
    name : str
        Name of the player.
    investment_set : np.ndarray
        Initial investment set of the player (i_min, i_max).
    benefit_function : callable
        Benefit function shape of the player, it takes as argument (GDP, i_max, percentage_green).
    GDP_initial : float
        Initial GDP of the player, i.e the maximum a player can attain with his benefit function.
    damage_function : callable
        Damage function, it takes as argument emissions.
    impact_factor_of_temperature : float
        Temperature multiplier of the player.
    alpha : float, optional
        Power of the damage function, by default 1.
    emission_saturation : float, optional
        Maximum possible emission level of the player, by default 1.
    percentage_green : float, optional
        Percentage of the benefit being decarbonated, by default 0.
    damage_in_percentage : bool, optional
        Specify if the damage is as percentage of GDP, by default True.
    discount : float, optional
        Discount rate for future utility, by default 1.

    Attributes
    ----------
    name : str
        Name of the player
    investment_set_initial : tuple
        Initial investment set of the player (i_min, i_max).
    GDP_initial : float
        Initial GDP of the player, i.e the maximum a player can attain with his benefit function.
    percentage_green : float
        Percentage of the benefit being decarbonated.
    benefit_shape : callable
        Benefit function shape of the player, it takes as argument (GDP, i_max, percentage_green).
    GDP_max : float
        Current GDP of the player, i.e the maximum a player can attain with his benefit function.
    investment_set : np.ndarray
        Current investment set of the player (i_min, i_max).
    T : int
        Number of time periods.
    damage_function : callable
        Damage function, it takes as argument emissions.
    delta : float
        Temperature multiplier of the player.
    alpha : float
        Power of the damage function.
    emission_saturation : float
        Maximum possible emission level of the player.
    damage_in_percentage : bool
        Specify if the damage is as percentage of GDP.
    discount : float
        Discount rate for future utility.
    """

    def __init__(self, name: str,
                investment_set: np.ndarray,
                benefit_function: Callable,
                GDP_initial: float,
                damage_function: Callable,
                impact_factor_of_temperature: float,
                scm : Simple_Climate_Model = SCM,
                alpha: float = 1,
                emission_saturation: float = 1,
                percentage_green: float = 0,
                damage_in_percentage: bool = True,
                discount: float = 1,
                state: float = 0) -> None:

        self.name: str = name
        "Name of the player"

        self.investment_set_initial: np.ndarray = investment_set
        "Initial investment set of the player (i_min, i_max)"

        self.GDP_initial: float = GDP_initial
        "Initial GDP of the player, i.e the maximum a player can attain with his benefit function."

        self.percentage_green: float = percentage_green
        "Percentage of the benefit being decarbonated."

        self.benefit_shape: Callable = benefit_function
        "Benefit function shape of the player, it takes as argument (GDP, i_max, percentage_green)."

        self.GDP_max = deepcopy(GDP_initial)
        "Current GDP of the player, i.e the maximum a player can attain with his benefit function."

        self.investment_set = deepcopy(investment_set)  # deepcopy for create a new memory location
        "Current investment set of the player (i_min, i_max)"

        self.T: int = investment_set.shape[0]
        "Number of time periods"

        self.damage_function = damage_function
        "Damage function, it takes as argument emissions."

        self.delta: float = impact_factor_of_temperature
        "Temperature multiplier the player"

        self.alpha: float = alpha
        "Power of the damage function"

        self.damage_in_percentage: bool = damage_in_percentage
        "Specify if the damage is as percentage of GDP."

        self.scm = deepcopy(scm)
        "Simple Climate Model of the player"

        self.discount: float = discount

        self.state: float = state
        "State variable used in the emission function"

        self.emission_saturation: Union[float, np.ndarray] = emission_saturation
        "Maximum possible emission level of the player"


    def reset_player(self):
        # deepcopy for reset due to pointer
        self.investment_set = deepcopy(self.investment_set_initial)
        self.GDP_max = deepcopy(self.GDP_initial)
        self.state = deepcopy(0)
        self.emission_saturation = deepcopy(self.emission_saturation)

    def reset(self):
        """Function which reset the player.
        """
        self.reset_player()

    def emission(self, investment: Union[float, np.ndarray], **kwargs) -> np.ndarray:
        """Calculate the player's emission based on their investment and the state variable using a decreasing sigmoid function.

        Parameters
        ----------
        investment : float
            The player's investment level.

        Returns
        -------
        float
            The player's emission level.
        """

        state = kwargs.pop('state', self.state)
        emission_saturation = kwargs.pop('emission_saturation', self.emission_saturation)


        new_state = deepcopy(state)
            
        if isinstance(investment, float):
            # Update state based on current investment and previous state
            self.state = 0.5 * self.state + investment
            return emission_saturation * (1 -  1 / (1 + np.exp(-self.state)))
        else:
            emissions = []
            for inv in investment:
                new_state = 0.5 * new_state + inv
                emissions.append(emission_saturation * (1 -  1 / (1 + np.exp(-self.state))))
            return np.array(emissions)

    def benefit(self, investments: Union[float,np.ndarray], **kwargs):
        GDP_max = kwargs.get('GDP_max', self.GDP_max)  # non-t dependent yet
        i_max = kwargs.get('i_max', self.investment_set[:, 1])  # t dependent
        percentage_green = kwargs.get('percentage_green', self.percentage_green)  # non-t dependent yet
        t = kwargs.get('t', None)
        if t is None:
            t0 = kwargs.get('t0', 0)
            return np.array([self.benefit_shape(GDP_max, i_max[t0+t], percentage_green)(investments[t]) for t in range(len(investments))])
        else:
            return self.benefit_shape(GDP_max, i_max[t], percentage_green)(investments)

    def damages(self, investment, sum_other_emissions, **kwargs):
        emissions = self.emission(investment, **kwargs) + sum_other_emissions
        temperature_target = kwargs.pop('temperature_target', None)
        final_multiplier = kwargs.pop('final_multiplier', FINAL_MULTIPLIER)

        atmospheric_temperatures = kwargs.get('temp', self.scm.evaluate_trajectory(emissions, **kwargs)[-1])
        if self.damage_in_percentage:
            GDP_max = kwargs.pop('GDP_max', self.GDP_max)
            damages = self.delta * GDP_max / 100 * self.damage_function(atmospheric_temperatures) ** self.alpha
        else:
            damages = self.delta * self.damage_function(atmospheric_temperatures) ** self.alpha

        if temperature_target is not None:
            damages = damages + final_multiplier * (atmospheric_temperatures[-1] - temperature_target)

        return damages

    def utility_one_shot(self, investment: float, sum_other_emissions: float, t: int, **kwargs) -> float:
        scm = kwargs.pop('scm', self.scm)
        temp = kwargs.pop('temp', scm.evaluate_trajectory(self.emission(investment) + sum_other_emissions), **kwargs)

        utility = self.benefit(investment, t=t, **kwargs) - self.damages(investment, sum_other_emissions, temp=temp, **kwargs)

        return utility

    def utility_sum_over_t(self, investments, sum_other_emissions, **kwargs):
        if self.discount != 1:
            discount = self.discount ** np.arange(len(investments))
        else:
            discount = 1
        benefit = self.benefit(investments, **kwargs)
        damages = self.damages(investments, sum_other_emissions, **kwargs)
        utility = np.sum(discount * (benefit - damages))

        return utility


    def best_response_over_t(self, others_emissions, **kwargs):
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', len(others_emissions)) + t0
        def response(x):
            return -self.utility_sum_over_t(x, others_emissions, **kwargs)
        action_set = self.investment_set[t0:tmax,:]
        bounds = Bounds(lb = action_set[:,0], ub = action_set[:,1], keep_feasible=True) 
        x0 = kwargs.get('x0', action_set[:,1])
        res = minimize(response, x0 = x0, bounds=bounds,  method='SLSQP', tol=1e-6, options={'maxiter': 10000, 'disp': 0})
        if not res.success:
            print(self.name)
            print(bounds)
            print(self.alpha)
            print(res)
        return res.x



class Player:
    """Class representing a player in a game of climate change.

    Parameters
    ----------
    name : str
        The name of the player.
    action_set : np.ndarray
        The initial action set of the player, i.e. the minimum and maximum emissions.
    benefit_function : callable
        The benefit function of the player, which takes as arguments the GDP, maximum emissions, and percentage green.
    GDP_initial : float
        The initial GDP of the player, i.e. the maximum attainable GDP with the benefit function.
    damage_function : callable
        The damage function of the player, which takes as an argument the emissions.
    impact_factor_of_temperature: float
        The temperature multiplier of the player.
    scm : Simple_Climate_Model, optional
        The Simple Climate Model of the player.
    alpha : float, optional
        The power of the damage function.
    increase_co2 : float, optional
        The ratio by which the player tends to increase its CO2 emissions over the game if it polluted at maximum.
    percentage_green : float, optional
        The percentage of the benefit being decarbonated.
    damage_in_percentage : bool, optional
        Specifies if the damage is a percentage of the GDP.
    discount : float, optional
        The discount factor used for calculating discounted utility.

    Attributes
    ----------
    name : str
        The name of the player.
    action_set_initial : tuple
        The initial action set of the player (e_min, e_max).
    GDP_initial : float
        The initial GDP of the player, i.e the maximum a player can attain with his benefit function.
    percentage_green : float
        The percentage of the benefit being decarbonated.
    benefit_shape : callable
        The benefit function shape of the player, it takes as argument (GDP, e_max, percentage_green).
    GDP_max : float
        The maximum GDP attainable by the player.
    action_set : np.ndarray
        The current action set of the player (e_min, e_max).
    T : int
        The number of time periods.
    scm : Simple_Climate_Model
        The Simple Climate Model of the player.
    damage_function : callable
        The damage function of the player, which takes as an argument the emissions.
    delta : float
        The temperature multiplier of the player.
    alpha : float
        The power of the damage function.
    increase_co2 : float
        The ratio by which the player tends to increase its CO2 emissions over the game if it polluted at maximum.
    damage_in_percentage : bool
        Specifies if the damage is a percentage of the GDP.
    discount : float
        The discount factor used for calculating discounted utility.
    """

    def __init__(self, name : str,
                action_set : np.ndarray,
                benefit_function : callable, 
                GDP_initial : float,
                damage_function : callable,
                impact_factor_of_temperature: float,
                scm : Simple_Climate_Model = SCM,
                alpha : float = 1,
                increase_co2 = INCREASE_CO2_RATIO,
                percentage_green = 0,
                damage_in_percentage = True,
                discount = 1) -> None:
        
        """
        Parameters:
        -----------
        name : str
            Name of the player.
        action_set : np.ndarray
            Initial action set of the player (e_min, e_max).
        benefit_function : callable
            Benefit function shape of the player, it takes as argument (GDP, e_max, percentage_green).
        GDP_initial : float
            Initial GDP of the player, i.e., the maximum a player can attain with his benefit function.
        damage_function : callable
            Damage function, it takes as argument emissions.
        impact_factor_of_temperature : float
            Temperature multiplier the player.
        scm : Simple_Climate_Model
            Simple Climate Model of the player.
        alpha : float, optional (default = 1)
            Power of the damage function.
        increase_co2 : float, optional (default = INCREASE_CO2_RATIO)
            Ratio of player tend to increase his co2 emissions over a game if it polluted at maximum.
        percentage_green : int, optional (default = 0)
            Percentage of the benefit being decarbonated.
        damage_in_percentage : bool, optional (default = True)
            Specify if the damage is as percentage of GDP.
        discount : float, optional (default = 1)
            Discount rate of the utility.

        Returns:
        --------
        None
        """

        self.name : str = name
        self.action_set_initial : tuple = action_set
        self.GDP_initial : float = GDP_initial
        self.percentage_green = percentage_green
        self.benefit_shape : callable = benefit_function
        self.GDP_max = deepcopy(GDP_initial)
        self.action_set = deepcopy(action_set)
        self.T : int = action_set.shape[0]
        self.scm = deepcopy(scm)
        self.damage_function = damage_function
        self.delta = impact_factor_of_temperature
        self.alpha = alpha
        self.increase_co2 = increase_co2
        self.damage_in_percentage : bool = damage_in_percentage
        self.discount = discount


    def reset_player(self):
        """Function which reset the player to the initial state for a given time period.

        Parameters
        ----------
        t : int
            The time period to reset the player for.
        """

        # deepcopy for reset due to pointer 
        self.action_set = deepcopy(self.action_set_initial)
        self.GDP_max = deepcopy(self.GDP_initial)


    def reset_scm(self) -> None:
        """Function which reset the SCM of the player.
        """
        self.scm.reset()


    def reset(self):
        """Function which reset the player and the SCM.
        """
        self.reset_player()
        self.reset_scm()

    def benefit(self, actions : np.ndarray, **kwargs):
        """
        Calculate the benefit of the player for a given set of actions.

        Parameters
        ----------
        actions : np.ndarray
            Array of emissions, one for each time period.

        Keyword Arguments
        -----------------
        GDP_max : float, optional
            Maximum GDP attainable of the player. Often it will be replaced with self.GDP_max but in some case it's useful to have it as a parameter.
        e_max : np.ndarray, optional
            Array of maximum emissions allowed by the player for each time period. If not provided, it will take the default values in self.action_set.

        Returns
        -------
        np.ndarray
            Array of benefits, one for each time period.
        """
        GDP_max = kwargs.get('GDP_max', self.GDP_max)
        e_max = kwargs.get('e_max', self.action_set[:,1])
        percentage_green = kwargs.get('percentage_green', self.percentage_green)
        t = kwargs.get('t', None)
        if t is None:
            t0 = kwargs.get('t0', 0)
            return np.array([self.benefit_shape(GDP_max, e_max[t0+t], percentage_green)(actions[t]) for t in range(len(actions))])
        else : 
            return self.benefit_shape(GDP_max, e_max[t], percentage_green)(actions) 

    def damages(self, emissions, temperature=None):
        """Calculate the damages caused by the player's emissions.

        Parameters
        ----------
        emissions : float or np.ndarray
            The player's emissions.

        temperature : float or np.ndarray, optional
            The temperature or temperature trajectory. If not provided, the player will generate a temperature with the given emissions using the SCM.

        Returns
        -------
        float or np.ndarray
            The damages caused by the player's emissions.
        """
        if isinstance(emissions, float):
            if temperature is None:
                temperature = self.scm.evaluate_trajectory(emissions)[-1]

            if self.damage_in_percentage:
                GDP_max = self.GDP_max
                damages = self.delta * GDP_max / 100 * (self.damage_function(temperature)) ** self.alpha
            else:
                damages = self.delta * (self.damage_function(temperature)) ** self.alpha
        else:
            atmospheric_temperatures = None
            if temperature is not None:
                atmospheric_temperatures = temperature[-1]
            else:
                atmospheric_temperatures = self.scm.evaluate_trajectory(emissions)[-1]

            if self.damage_in_percentage:
                GDP_max = self.GDP_max
                damages = self.delta * GDP_max / 100 * self.damage_function(atmospheric_temperatures) ** self.alpha
            else:
                damages = self.delta * self.damage_function(atmospheric_temperatures) ** self.alpha

        return damages