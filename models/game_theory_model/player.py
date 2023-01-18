"""A player class.

Here we create a player class where wa concatenate characteristics of a player:

- Action set
- Utility function
- etc..

"""

import numpy as np
from scipy.optimize import *
from ..geophysic_models import *
from copy import deepcopy
from parameters import *
from .jacobian import * 

class player_class:
    """Class of a player.
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
                

        self.name : str = name
        "Name of the player"

        self.action_set_initial : tuple = action_set
        "Initial action set of the player (e_min, e_max)"

        self.GDP_initial : float = GDP_initial
        "Initial GDP of the player, i.e the maximum a player can attain with his benefit function."

        self.percentage_green = percentage_green
        "Percentage of the benefit being decarbonated."
        
        self.benefit_shape : callable = benefit_function
        "Benefit function shape of the player, it takes as argument (GDP, e_max, percentage_green)."

        self.GDP_max = deepcopy(GDP_initial)
        "Current GDP of the player, i.e the maximum a player can attain with his benefit function."

        self.action_set = deepcopy(action_set)   # deepcopy for create a new memory location
        "Current action set of the player (e_min, e_max)"

        self.scm = deepcopy(scm)
        "Simple Climate Model of the player"


        self.benefit_function = self.benefit_shape(self.GDP_max, self.action_set[1], percentage_green)
        "Current Benefit function of the player, it takes as argument emissions."

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
        """Function which reset the player to the inital state.
        """

        # deepcopy for reset due to pointer 
        self.action_set = deepcopy(self.action_set_initial)
        self.GDP_max = deepcopy(self.GDP_initial)
        self.benefit_function = self.benefit_shape(deepcopy(self.GDP_initial), deepcopy(self.action_set_initial[1]), deepcopy(self.percentage_green))


    def reset_scm(self) -> None:
        """Function which reset the SCM of the player.
        """
        self.scm.reset()


    def reset(self):
        """Function which reset the player and the SCM.
        """
        self.reset_player()
        self.reset_scm()


    def utility_one_shot(self, action : float, sum_other_actions : float, **kwargs) -> float :# temp : float = None, scm = False) -> float:
        """Utility of the player.

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

        if self.damage_in_percentage:
            kwargs['GDP_max'] = kwargs.get('GDP_max',self.GDP_max)
        else:
            kwargs.pop('GDP_max', None)
        scm = kwargs.pop('scm', self.scm)
        temp = kwargs.pop('temp', None)
        
        return self.benefit_function(action) - self.delta * (self.damage_function(temp= temp, sum_action = action + sum_other_actions, scm=scm, **kwargs))**self.alpha 

    def utility_sum_over_t(self, emissions, others_emissions, **kwargs):
        if self.discount !=1:
            discount = self.discount**np.arange(len(emissions))
        else:
            discount = 1
        all_emissions = emissions + others_emissions
        atmospheric_temperatures = self.scm.evaluate_trajectory(all_emissions, **kwargs)[-1]
        damages = self.delta  * self.damage_function(atmospheric_temperatures)
        benefit = self.benefit_function(emissions)
        return np.sum(discount * (benefit - damages))

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
            jac_damage = jacobian_damage_function(temperature_AT, self.damage_function)

            jac_benefit = np.array([derivative(self.benefit_function,x[t], order=5, dx=1e-6) for t in range(t_periode)])
            return -discount * (jac_benefit - self.delta * jac_damage @ jac_temperature_AT @ jac_forcing @ jac_carbon_AT)
        return jacobian

    def best_response_over_t(self, others_emissions, **kwargs):
        def response(x):
            return -self.utility_sum_over_t(x, others_emissions, **kwargs)
        jac= self.jacobian_utility_sum_over_t(others_emissions, **kwargs)
        repeated_action_set = np.tile(self.action_set, (len(others_emissions),1))
        bounds = Bounds(lb = repeated_action_set[:,0], ub=repeated_action_set[:,1], keep_feasible=True) 
        x0 = kwargs.get('x0', repeated_action_set[:,0])

        res = minimize(response, x0 = x0, bounds=bounds,  tol = 1e-5, options={'maxiter': 10000, 'disp': 0})
        if not res.success:
            print(res.success)
        return res.x




    def utility_n_shot(self, n: int, action : np.ndarray, sum_other_actions : np.ndarray) -> float:
        """IN PROGRESS Utility of the player.

        Parameters
        ----------
        action : float
            Emission of the player.
        sum_other_actions : float
            Emissions of other players.
        GDP_max : float
            Maximum GDP attainable of the player. Often it will be remplace with self.GDP_max but in some case it's usefull to have it as a parameter.
        temp : float, optional
            Current temperature. If not provided, the player will generate a temperature with the given actions with the SCM. If given, it speed up the calculs, by default None
        scm : bool, optional
            SCM used by the player. If not provided, the player will generate a temperature with the given actions with this specific SCM, by default False

        Returns
        -------
        float
            _description_
        """
        action_set = deepcopy(self.action_set)
        GDP_max = deepcopy(self.GDP_max)
        benefit_function = deepcopy(self.benefit_function)
        scm = deepcopy(self.scm)
        utility = 0
        for k in range(n):
            utility =+ benefit_function(action[k]) - self.delta * (self.damage_function(action + sum_other_actions, scm, GDP_max))**self.alpha
            action_set, GDP_max, benefit_function = self.simulate_update_player()

        return utility



    def best_response_one_shot(self, sum_other_actions : float) -> float:
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
            return -self.utility_one_shot(x, sum_other_actions)

        res = minimize_scalar(response, bounds=self.action_set, method='bounded', options={'xatol': 1e-05, 'maxiter': 1000, 'disp': 0})
        return res.x

    def update_scm(self, sum_actions : float, exogeneous_emission : float, exogeneous_radiative_forcing) -> None:
        self.scm.five_years_cycle_deep(sum_actions, exogeneous_emission, exogeneous_radiative_forcing)


        
    def update_player(self, previous_emission : float) -> None:
        """Function which update the player from a previous state for a given previous emission.

        Notes
        -----

        Main place for improvement.

        Parameters
        ----------
        previous_emission : float
            Emission of the players in the previous iteration of the game. 
        """
        # We calculate the ratio of previous emission compare to what the player could have poluted.
        ratio = previous_emission / self.action_set[1]
        # The action set of the player update 
        self.action_set[1] =  self.action_set[1] * (1 + ratio * self.increase_co2)
        self.GDP_max = self.GDP_max * (1 + ratio*0.02)
        self.benefit_function = self.benefit_shape(self.GDP_max, self.action_set[1], self.percentage_green)

    def simulate_update_player(self, previous_emission : float, action_set : np.ndarray, GDP_max : float,
                     percentage_green):
        """IN PROGRESS

        Parameters
        ----------
        previous_emission : float
            _description_
        action_set : np.ndarray
            _description_
        GDP_max : float
            _description_
        percentage_green : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        # We calculate the ratio of previous emission compare to what the player could have poluted.
        ratio = previous_emission / action_set[1]
        # The action set of the player update 
        action_set[1] =  action_set[1] * (1 + ratio * self.increase_co2)
        GDP_max = GDP_max * (1 + ratio*0.002)
        benefit_function = self.benefit_shape(GDP_max, action_set[1], percentage_green)
        return action_set, GDP_max, benefit_function

