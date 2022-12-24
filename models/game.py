"""Test.

"""

import numpy as np
from scipy.optimize import *
from .geophysic_models import *
from .game_theory_model.player import player_class
from copy import deepcopy
from parameters import *




def create_players(list_of_names : list[str] = NAMES,
                    list_action_sets : list[tuple] = ACTION_SETS,
                    list_benefit_functions : list[callable] = BENEFITS,
                    list_GDP_initial : list[float] = GDP_MAX,
                    damage_function : callable = DAMAGE,
                    alpha : float = ALPHA,
                    list_deltas : list[float] = DELTAS,
                    list_coef_increase_co2 : list[float] = INCREASE_COEF_CO2_RATIO,
                    list_percentage_green : list[float] = PERCENTAGES_GREEN,
                    damage_in_percentage : bool = PERCENTAGE_GDP):
    """Function which generate a list of players.

    Parameters
    ----------
    list_of_names : list[str], optional
        The list of name for the players, by default NAMES
    list_action_sets : list[tuple], optional
        The list of action set (e_min, ,e_max) for the players, by default ACTION_SETS
    list_benefit_functions : list[callable], optional
        The list of benefit function shape for the player, by default BENEFITS
    list_GDP_initial : list[float], optional
        The list of inital maximum GDP for the players, by default GDP_MAX
    damage_function : callable, optional
        The damage function players will have. It's the same for every players, by default DAMAGE
    alpha : float, optional
        Power of the damage function, by default ALPHA
    list_deltas : list[float], optional
        The list of the temperature multiplier for the players, by default DELTAS
    list_coef_increase_co2 : list[float], optional
        The list of the coefficient for increasing emission after playing the game one, by default INCREASE_COEF_CO2_RATIO
    list_percentage_green : list[float], optional
        The list of the percentage of the benefits of players being decarbonated, by default PERCENTAGES_GREEN

    Returns
    -------
    list
        A list of players.
    """

    return [player_class(name = list_of_names[i],
                          action_set = list_action_sets[i],
                          benefit_function = list_benefit_functions[i],
                          GDP_initial = list_GDP_initial[i],
                          damage_function = damage_function,
                          impact_factor_of_temperature = list_deltas[i],
                          increase_co2 = list_coef_increase_co2[i],
                          percentage_green=list_percentage_green[i],
                          alpha=alpha,
                          damage_in_percentage = damage_in_percentage) for i in range(N)]

class Game:
    """Test
    """

    def __init__(self, 
                list_players : list[player_class],
                scm : Simple_Climate_Model = SCM,
                exogeneous_emission : np.ndarray = E_EX,
                exogeneous_radiative_forcing : np.ndarray = F_EX,
                horizon : int = FINAL_YEAR,
                initial_time : int = FIRST_YEAR,
                time_step : int = 5 ) -> None:

        self.N : int = len(list_players) 
        "Number of player in the game"      
        
        self.T : int = int((horizon - initial_time)/time_step)+1
        "Number of time the game is repeated"
        # Business As Usual

        self.strat_action_profiles = np.zeros((self.N,self.T))
        self.strat_sum_action_profiles = np.zeros(self.T)
        self.strat_utilities_profiles = np.zeros((self.N, self.T))
        self.strat_sum_utilities_profiles = np.zeros(self.T)
        self.strat_temp_profile = np.zeros(self.T)


        # Simple Climate Model

        self.scm_game : Simple_Climate_Model = deepcopy(scm)
        "Simple eClimate model of the game."

        self.ex_e : np.ndarray = exogeneous_emission
        "Exogeneous emission during the repetition of the game."
        self.ex_f : np.ndarray = exogeneous_radiative_forcing
        "Exogeneous radiative forcing during the repetion of the game."
        # Game Properties 

        self.list_players : list[player_class] = deepcopy(list_players)
        "List of player in the game with their characteristics."

    def reset(self) :
        """Function which reset the game. i.e. It reset the SCM of the game, the SCM of the players and the player themselve. 
        """
        self.scm_game.reset()
        for player in self.list_players:
            player.reset_scm()
            player.reset()


    def update_player(self, actions : np.ndarray):
        """Function which update the player with respect to their previous actions.

        Parameters
        ----------
        actions : np.ndarray
            Previous actions of the players.
        """
        for idx, player in enumerate(self.list_players):
            player.update_player(actions[idx])


    def update_players_scm(self):
        """Function that update the SCM of players with the current states of the game SCM.
        """
        for player in self.list_players:
            player.scm.initialize(deepcopy(self.scm_game.carbon_state), deepcopy(self.scm_game.temperature_state), deepcopy(self.scm_game.num_cycle))

    
    def get_players_utilities(self):
        """Function which extract utility functions of the players. 

        Returns
        -------
        list(callable)
            List of the utility functions of the players of the game at current state.
        """
        fcts = []
        for player in self.list_players:
            fcts.append(deepcopy(player.utility_one_shot))
        return fcts


    def get_action_space(self) -> np.ndarray :
        """Function which extract the action sets of the players of the game at current state.

        Returns
        -------
        np.ndarray
            Array describing the action set :math:`\mathcal{A}` of the game. 
        """
        action_space = np.zeros((self.N,2))
        for idx, player in enumerate(self.list_players):
            action_space[idx] = deepcopy(player.action_set)
        return action_space

    def get_GDP_max(self) -> np.ndarray :
        """Function which extract the maximum GDP of the players of the game at current state.

        Returns
        -------
        np.ndarray
            Array of GDP_max.
        """
        GDP_max = np.zeros(self.N)
        for idx, player in enumerate(self.list_players):
            GDP_max[idx] = deepcopy(player.GDP_max)
        return GDP_max

    def repeated_one_shot_game(self, rule : callable):
        """Function which simulate repetition of the one shot game for a given solution concept. For example,
        Nash equilibria, Social optimum,...

        Parameters
        ----------
        rule : callable
            The solution concept used.

        Returns
        -------
        a_p : np.ndarray 
            Actions profile of the players allong the game. A (N,T) array.
        sum_a_p : np.ndarray
            Sum of the actions profile of the players allong the game. A (T) array.
        u_val_p : np.ndarray
            Value of utilities of the players allong the game. A (N,T) array.
        u_fct_p : list(callable)
            Utility functions of the players allong the game. A list of lenght T where each element is a list of lenght N of callable.
            u_fct_p[t][n] is the utility function of player n at time t. 
        sum_u_val_p : np.ndarray
            Sum of the utilities of the players allong the game. A (T) array.
        a_space_p : np.ndarray
            Actions space of the players allong the game. A (N,T) array.
        temp_p : np.ndarry
            Global mean atmospheric temperature profile allong the game. A (T) array.

        """

        # Creation of the futures return.

        a_p = np.zeros((self.N, self.T)) 
        sum_a_p = np.zeros(self.T)
        u_val_p = np.zeros((self.N, self.T))
        u_fct_p = [None] * self.T
        a_space_p = np.zeros((self.N,self.T,2))
        gdp_max_p = np.zeros((self.N, self.T))
        temp_p = np.zeros(self.T)

        # Reset the game: SCM, players' utility and action sets.
        self.reset()

        # Loop of the games.
        for time in range(self.T):
            ## Initialization of the One-Shot game.

            # Players get the SCM values from the SCM of the One-game. 
            self.update_players_scm()


            ## Realization of the game.

            # Players choose their actions following the given rule.
            a_p[:,time] = rule()

            # Actions are summed 
            sum_a_p[time] = np.sum(a_p[:,time])

            # We save the utility functions of the players for the current game.
            u_fct_p[time] = self.get_players_utilities()

            # We save the actions space of the players for the current game.
            a_space_p[:,time,:] = self.get_action_space() 

            # We save the GDP_max of the players for the current game.
            gdp_max_p[:,time] = self.get_GDP_max()


            ## Updating of the players.

            # The players utility and action set get update with respect to the previous action
            self.update_player(a_p[:,time])

            ## Climate modelization take place

            # Players' emissions are injected in the SCM.
            self.scm_game.five_years_cycle_deep(sum_a_p[time], self.ex_e[time], self.ex_f[time])

            # We save the temperature of the game at current state.
            temp_p[time] = self.scm_game.atmospheric_temp

        # Calculus of the utilities values for all games.
        for time in range(self.T):
            for indice in range(N):
                u_val_p[indice,time] = u_fct_p[time][indice](a_p[indice,time], sum_a_p[time] - a_p[indice,time], GDP_max = gdp_max_p[indice,time], temp=temp_p[time])

        # Sum of the utilities values for all games.
        sum_u_val_p = np.sum(u_val_p, axis=0)

        return a_p, sum_a_p, u_val_p, u_fct_p, sum_u_val_p, a_space_p, temp_p



    def best_response_dynamic_one_shot(self, step_criterion : int = 100, utility_criterion : float = 0.0001):
        """Best Response Dynamic function which allow us to find Nash equilibria.

        Parameters
        ----------
        step_criterion : int, optional
            Maximum number of iteration of the Best Response Dynamic before stoping, by default 100
        utility_criterion : float, optional
            Minimal difference between two step we need to obtain before stoping, by default 0.0001

        Returns
        -------
        np.ndarray
            End point of the BRD.
        """

        #Const
        loss = 1000
        k = 0
        
        ## Initialization of the BRD
        # Initial action put at 0.
        inital_action = np.zeros(self.N)
        # Creation of the list of Best response actions.
        list_of_all_action = [inital_action]

        while (loss > utility_criterion) and (k < step_criterion):

            list_action = deepcopy(list_of_all_action[-1])
            for indice in range(self.N):
                # Best response fonction of a players is a function of the sum of other actions.
                # Sum of others players actions
                sum_other_actions = np.sum(list_action) - list_action[indice]

                # Calculate the Best Responce of the player indice for the given sum of other players' emissions.
                list_action[indice] = self.list_players[indice].best_response_one_shot(sum_other_actions)
                # Previously 
                # list_action[indice] = round(self.list_players[indice].best_response_one_shot(sum_other_actions), 5)

            #Increase in the iteration of the BRD
            k +=1
            # Calcul of the improvement attained for the this iteration of the BRs
            loss = sum([(list_action[indice] - list_of_all_action[-1][indice])**2 for indice in range(self.N)])
            # Save the actual point of the the BRs
            list_of_all_action.append(list_action)

        return list_of_all_action[-1]

    def one_shot_game(self, scm : Simple_Climate_Model)-> None:
        utilities = np.zeros(self.N)
        for indice in range(self.N):
            player = self.list_players[indice]
            player.reset()
            player.scm.initialize(deepcopy(scm.carbon_state), deepcopy(scm.temperature_state), scm.num_cycle)
            
        actions = self.best_response_dynamic_one_shot()
        sum_action = np.sum(actions)

        temp = scm.five_years_atmospheric_temp(sum_action)

        for indice in range(self.N):
            player = self.list_players[indice]
            utilities[indice] = player.utility_one_shot(actions[indice], sum_action-actions[indice], temp=temp)
        
        sum_utilities = np.sum(utilities)

        return actions, utilities, sum_action, sum_utilities

    def repeated_one_shot_game_NE(self)-> None:

        self.ne_a_p, self.ne_sum_a_p, self.ne_u_p, self.ne_u_fct_p, self.ne_sum_u_p, self.ne_a_space_p, self.ne_temp_p = self.repeated_one_shot_game(self.best_response_dynamic_one_shot)
    
    
    def business_as_usual(self):

        action_space = self.get_action_space()
        return action_space[:,1]
    

    def repeated_one_shot_game_BAU(self)-> None:
 
        self.bau_a_p, self.bau_sum_a_p, self.bau_u_p, self.bau_u_fct_p, self.bau_sum_u_p, self.bau_a_space_p, self.bau_temp_p = self.repeated_one_shot_game(self.business_as_usual)


    def fct_sum_utilities_one_shot(self, actions : np.ndarray) -> float:

        sum_actions = np.sum(actions)
        sum_utilities = 0

        for idx, player in enumerate(self.list_players) :
            sum_utilities = sum_utilities + player.utility_one_shot(actions[idx], sum_actions - actions[idx])
        return sum_utilities


    def social_optimum_one_shot(self) -> np.ndarray:

        bounds = self.get_action_space()
        boundaries = Bounds(lb=bounds[:,0], ub=bounds[:,1], keep_feasible=True)

        def fct_social_optimum(x : np.ndarray):
            return -self.fct_sum_utilities_one_shot(x)

        res = minimize(fct_social_optimum, x0=bounds[:,1], bounds=boundaries)

        return res.x
    

    def repeated_one_shot_game_SO(self)-> None:
        self.so_a_p, self.so_sum_a_p, self.so_u_p, self.so_u_fct_p, self.so_sum_u_p, self.so_a_space_p, self.so_temp_p = self.repeated_one_shot_game(self.social_optimum_one_shot)

                                                             
        
    def repeated_one_shot_game_with_strategies_profile(self, array_action_profile : np.ndarray) -> None:
        self.reset()
        
        self.strat_action_profiles = array_action_profile
        self.strat_sum_action_profiles = np.sum(array_action_profile,axis=0)

        for time in range(self.T):
            self.scm_game.five_years_cycle_deep(self.strat_sum_action_profiles[time])
            self.strat_temp_profile[time] = self.scm_game.atmospheric_temp

        for indice in range(self.N):
            player = self.list_players[indice]
            self.strat_utilities_profiles[indice] = player.utility_one_shot(self.strat_action_profiles[indice],
                                                                     self.strat_sum_action_profiles-self.strat_action_profiles[indice], self.strat_temp_profile)
                                                                     
        self.strat_sum_utilities_profiles = np.sum(self.strat_utilities_profiles,axis = 0)


    def one_shot_game_with_strategies_profile(self, array_action : np.ndarray, sum_action: float, scm : Simple_Climate_Model) -> None:
        utilities = np.zeros(self.N)

        temp = scm.five_years_atmospheric_temp(sum_action)

        for indice in range(self.N):
            player = self.list_players[indice]
            utilities[indice] = player.utility_one_shot(array_action[indice],
                                                                     sum_action - array_action[indice], temp)
                                                                     

        return utilities

