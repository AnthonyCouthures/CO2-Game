"""Test.

"""


import numpy as np
from scipy.optimize import Bounds,minimize, NonlinearConstraint
from .geophysic_models import *
from .game_theory_model.player import player_class
from .game_theory_model.jacobian import * 
from copy import deepcopy
from parameters import *
from functools import partial
from tqdm import tqdm
from collections.abc import Callable
from tabulate import tabulate
import time



def create_players(list_of_names : list[str] = NAMES,
                    list_action_sets : np.ndarray = ACTION_SETS,
                    list_benefit_functions : list[Callable] = BENEFITS,
                    list_GDP_initial : np.ndarray = GDP_MAX,
                    damage_function : Callable = DAMAGE,
                    alpha : float = ALPHA,
                    list_deltas : np.ndarray = DELTAS,
                    list_coef_increase_co2 : list[float] = INCREASE_COEF_CO2_RATIO,
                    list_percentage_green : list[float] = PERCENTAGES_GREEN,
                    damage_in_percentage : bool = PERCENTAGE_GDP,
                    discount = DISCOUNT):
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

    return [player_class(name = deepcopy(list_of_names[i]),
                          action_set = deepcopy(list_action_sets[i]),
                          benefit_function = deepcopy(list_benefit_functions[i]),
                          GDP_initial = deepcopy(list_GDP_initial[i]),
                          damage_function = damage_function,
                          impact_factor_of_temperature = deepcopy(list_deltas[i]),
                          increase_co2 = deepcopy(list_coef_increase_co2[i]),
                          percentage_green = deepcopy(list_percentage_green[i]),
                          alpha = alpha,
                          damage_in_percentage = damage_in_percentage, 
                          discount = discount) for i in range(len(list_of_names))]



class GameResults:
    def __init__(self):
        self.data = {
            'action': {
                'ne': {'oneshot': [], 'planning': {}, 'receding': {}},
                'so': {'oneshot': [], 'planning': {}, 'receding': {}}
            },
            'sum_action': {
                'ne': {'oneshot': [], 'planning': {}, 'receding': {}},
                'so': {'oneshot': [], 'planning': {}, 'receding': {}}
            },
            'utility': {
                'ne': {'oneshot': [], 'planning': {}, 'receding': {}},
                'so': {'oneshot': [], 'planning': {}, 'receding': {}}
            },
            'sum_utility': {
                'ne': {'oneshot': [], 'planning': {}, 'receding': {}},
                'so': {'oneshot': [], 'planning': {}, 'receding': {}}
            },
            'temp': {
                'ne': {'oneshot': [], 'planning': {}, 'receding': {}},
                'so': {'oneshot': [], 'planning': {}, 'receding': {}}
            }
        }

    def store(self, category, case, subcase, value, time_duration=None):
        if time_duration and (subcase == 'planning' or subcase == 'receding'):
            # If time_duration doesn't exist yet, initialize it
            if time_duration not in self.data[category][case][subcase]:
                self.data[category][case][subcase][time_duration] = []
            self.data[category][case][subcase][time_duration].append(value)
        else:
            self.data[category][case][subcase].append(value)



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
                time_step : int = 5,
                temperature_target = None,
                final_multiplier = FINAL_MULTIPLIER,
                update_allowed = False) -> None:

        self.N : int = len(list_players) 
        "Number of player in the game"      
        self.horizon = horizon

        self.temperature_target = temperature_target
        self.final_multiplier = final_multiplier
        self.T : int = int((horizon - initial_time)/time_step)
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

        self.list_damage_function = [player.damages for player in self.list_players]

        self.update_allowed = update_allowed

        self.deltas = np.array([player.delta for player in list_players])

    def reset(self) :
        """Function which reset the game. i.e. It reset the SCM of the game, the SCM of the players and the player themselve. 
        """
        self.scm_game.reset()
        for player in self.list_players:
            player.reset_scm()
            player.reset()


    def update_player(self, actions : np.ndarray, t : int):
        """Function which update the player with respect to their previous actions.

        Parameters
        ----------
        actions : np.ndarray
            Previous actions of the players.
        """
        for idx, player in enumerate(self.list_players):
            player.update_player(actions[idx], t)


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


    def get_action_space_at_t(self, t : int) -> np.ndarray :
        """Function which extract the action sets of the players of the game at current state.

        Returns
        -------
        np.ndarray
            Array describing the action set :math:`\mathcal{A}` of the game. 
        """
        action_space = np.zeros((self.N,2))
        for idx, player in enumerate(self.list_players):
            action_space[idx] = deepcopy(player.action_set[t])
        return action_space
    
    def get_action_space(self, **kwargs) -> np.ndarray :
        """Function which extract the action sets of the players of the game at current state.

        Returns
        -------
        np.ndarray
            Array describing the action set :math:`\mathcal{A}` of the game. 
        """
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', self.T) + t0
        action_space = np.zeros((self.N, tmax, 2))
        for idx, player in enumerate(self.list_players):
            action_space[idx, t0:tmax] = deepcopy(player.action_set[t0:tmax])
        return action_space
    
    def get_action_space_specific(self, indices: tuple[int, int], **kwargs) -> np.ndarray :
        """Function which extract the action sets of the players of the game at current state.

        Returns
        -------
        np.ndarray
            Array describing the action set :math:`\mathcal{A}` of the game. 
        """
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', self.T) + t0
        action_space = np.zeros((len(indices), tmax, 2))
        for idx in indices:
            player = self.list_players[idx]
            action_space[idx, t0:tmax] = deepcopy(player.action_set[t0:tmax])
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

    def repeated_one_shot_game(self, rule : Callable, **kwargs):
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
        u_p = np.zeros((self.N, self.T))
        temperature_AT = np.zeros(self.T)
        # Reset the game: SCM, players' utility and action sets.
        self.reset()
        # Loop of the games.
        for time in tqdm(range(self.T), desc='Repeated game'):
            ## Initialization of the One-Shot game.
            # Players get the SCM values from the SCM of the One-game. 
            self.update_players_scm()
            ## Realization of the game.
            # Players choose their actions following the given rule.
            a_p[:,time] = rule(t=time)
            # Actions are summed 
            sum_a_p[time] = np.sum(a_p[:,time])
            if self.update_allowed :
                ## Updating of the players.
                # The players utility and action set get update with respect to the previous action
                self.update_player(a_p[:,:time], time)
            ## Climate modelization take place
            # Players' emissions are injected in the SCM.
            self.scm_game.five_years_cycle_deep(sum_a_p[time]) #, self.ex_e[time], self.ex_f[time])
            # We save the temperature of the game at current state.
            temperature_AT[time] = self.scm_game.atmospheric_temp
        # Calculus of the utilities values for all games.
        for idx, player in enumerate(self.list_players):
            u_p[idx] = player.benefit(a_p[idx], **kwargs) - player.damages(sum_a_p, temp = temperature_AT, **kwargs)

        # Sum of the utilities values for all games.
        sum_u_p = np.sum(u_p, axis=0)

        return a_p, sum_a_p, u_p, sum_u_p, temperature_AT
    
    def best_response_dynamic_one_shot(self, t, step_criterion : int = 100, utility_criterion : float = 1e-10):
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
                list_action[indice] = self.list_players[indice].best_response_one_shot(sum_other_actions, t)
                # Previously 
                # list_action[indice] = round(self.list_players[indice].best_response_one_shot(sum_other_actions), 5)

            #Increase in the iteration of the BRD
            k +=1
            # Calcul of the improvement attained for the this iteration of the BRs
            loss = sum([(list_action[indice] - list_of_all_action[-1][indice])**2 for indice in range(self.N)])
            # Save the actual point of the the BRs
            list_of_all_action.append(list_action)

        return list_of_all_action[-1]

    def one_shot_game(self, t : int)-> tuple:
        utilities = np.zeros(self.N)
            
        actions = self.best_response_dynamic_one_shot(t)
        sum_action = np.sum(actions)


        for indice in range(self.N):
            player = self.list_players[indice]
            utilities[indice] = player.utility_one_shot(actions[indice], sum_action-actions[indice], t)
        
        sum_utilities = np.sum(utilities)

        return actions,  sum_action, utilities, sum_utilities
    
    def one_shot_game_NE(self, t=0) -> None:
        self.ne_a_p, self.ne_sum_a_p, self.ne_u_p, self.ne_sum_u_p = self.one_shot_game(t)

   
    def business_as_usual(self, t : int):
        action_space = self.get_action_space_at_t(t = t)
        return action_space[:,t,1]
    
    def repeated_one_shot_game_BAU(self)-> None:
 
        self.bau_a_p, self.bau_sum_a_p, self.bau_u_p, self.bau_sum_u_p, self.bau_temp_p = self.repeated_one_shot_game(self.business_as_usual)

    def fct_sum_utilities_one_shot(self, actions : np.ndarray, t : int) -> float:
        sum_actions = np.array(np.sum(actions))
        sum_utilities = 0
        for idx, player in enumerate(self.list_players) :
            sum_utilities = sum_utilities + player.utility_one_shot(np.array(actions[idx]), sum_actions - actions[idx], t)
        return sum_utilities

    def social_optimum_one_shot(self, t : int) -> np.ndarray:
        bounds = self.get_action_space_at_t(t = t)
        boundaries = Bounds(lb=bounds[:,0], ub=bounds[:,1], keep_feasible=True)
        def fct_social_optimum(x : np.ndarray):
            return -self.fct_sum_utilities_one_shot(x, t)
        res = minimize(fct_social_optimum, x0=bounds[:,1], bounds=boundaries)
        return res.x

    def repeated_one_shot_game_with_strategies_profile(self, array_action_profile : np.ndarray, t : int) -> None:
        self.reset()
        
        self.strat_action_profiles = array_action_profile
        self.strat_sum_action_profiles = np.sum(array_action_profile,axis=0)

        for time in range(self.T):
            self.scm_game.five_years_cycle_deep(self.strat_sum_action_profiles[time])
            self.strat_temp_profile[time] = self.scm_game.atmospheric_temp

        for indice in range(self.N):
            player = self.list_players[indice]
            self.strat_utilities_profiles[indice] = player.utility_one_shot(self.strat_action_profiles[indice],
                                                                     self.strat_sum_action_profiles-self.strat_action_profiles[indice], t, temp=self.strat_temp_profile)                                              
        self.strat_sum_utilities_profiles = np.sum(self.strat_utilities_profiles,axis = 0)

    def one_shot_game_with_strategies_profile(self, array_action : np.ndarray, sum_action: float) -> np.ndarray:
        utilities = np.zeros(self.N)

        # temp = self.scm_game.five_years_atmospheric_temp(sum_action)

        for indice in range(self.N):
            player = self.list_players[indice]
            utilities[indice] = player.utility_sum_over_t(array_action[indice],
                                                                     sum_action - array_action[indice], t=0)
        return utilities
    
    def game_with_strategies_profile(self, array_action : np.ndarray, sum_action: np.ndarray, **kwargs) -> tuple:
        utilities = np.zeros(self.N)

        temp = self.scm_game.evaluate_trajectory(sum_action, **kwargs)[-1]

        for indice in range(self.N):
            player = self.list_players[indice]
            utilities[indice] = player.utility_sum_over_t(array_action[indice],
                                                                     sum_action - array_action[indice],
                                                                     temp = temp)
        return utilities, temp

    def get_players_planning_utilities(self):
        """Function which extract utility functions of the players. 

        Returns
        -------
        list(callable)
            List of the utility functions of the players of the game at current state.
        """
        fcts = []
        for player in self.list_players:
            fcts.append(deepcopy(player.utility_sum_over_t))
        return fcts

    def planning_game(self, rule : Callable,  **kwargs):

        tmax = kwargs.get('tmax', self.T)
        # Creation of the futures return.

        a_p = np.zeros((self.N, tmax)) 
        u_p = np.zeros((self.N, tmax))
        # Reset the game: SCM, players' utility and action sets.
        ## Initialization of the planning game.
        # Players get the SCM values from the SCM of the planning game. 
        # self.update_players_scm()
        ## Realization of the game.
        # Players choose their actions following the given rule.
        a_p = rule(**kwargs)
        # Actions are summed 
        sum_a_p = np.sum(a_p, axis=0)
        ## Climate modelization 
        # Players' emissions are injected in the SCM.
        temperature_AT =  self.scm_game.evaluate_trajectory(sum_a_p, **kwargs)[-1]

        # Calculus of the utilities values for all games.

        for idx, player in enumerate(self.list_players):
            u_p[idx] = player.benefit(a_p[idx], **kwargs) - player.damages(sum_a_p, temp = temperature_AT, **kwargs)
        # Sum of the utilities values for all games.
        sum_u_p = np.sum(u_p, axis=0)

        return a_p, sum_a_p, u_p, sum_u_p, temperature_AT

    def planning_gradient_descent(self, **kwargs):
        self.ne_a_planning_gd, self.ne_sum_a_planning_gd, self.ne_u_planning_gd, self.ne_sum_u_planning_gd, self.ne_temp_planning_gd = self.planning_game(self.gradient_over_potential, **kwargs)


    def best_response_dynamic_planning(self, step_criterion : int = 1000, utility_criterion : float = 1e-6, **kwargs):
        """Best Response Dynamic function which allow us to find Nash equilibria for the planning.

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
        
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', self.T) + t0

        ## Initialization of the BRDhb
        action_space = self.get_action_space(**kwargs)
        upper_bounds = action_space[:,t0:tmax,1]/2
        # Initial action put at 0.
        inital_action = kwargs.pop('x0', upper_bounds)
        # Creation of the list of Best response actions.
        list_of_all_action = [inital_action]

        while (loss > utility_criterion) and (k < step_criterion):

            list_action = deepcopy(list_of_all_action[-1])
            for indice in range(self.N):
                # Best response fonction of a players is a function of the sum of other actions.
                # Sum of others players actions
                sum_other_actions = np.sum(list_action, axis=0) - list_action[indice]
                # print('sum_other_actions', sum_other_actions)
                # print('list_action[indice]', list_action[:,indice])

                # Calculate the Best Responce of the player indice for the given sum of other players' emissions.
                list_action[indice] = self.list_players[indice].best_response_over_t(sum_other_actions, x0 = list_action[indice], **kwargs)
                
            #Increase in the iteration of the BRD
            k +=1
            # Calcul of the improvement attained for the this iteration of the BRs
            loss = np.sum([(list_action[indice] - list_of_all_action[-1][indice])**2 for indice in range(self.N)])
            # Save the actual point of the the BRs
            list_of_all_action.append(list_action)
        return list_of_all_action[-1]

    def planning_BRD(self, **kwargs):
        self.ne_a_planning_brd, self.ne_sum_a_planning_brd, self.ne_u_planning_brd, self.ne_sum_u_planning_brd, self.ne_temp_planning_brd = self.planning_game(
            self.best_response_dynamic_planning, temperature_target = self.temperature_target, final_multiplier = self.final_multiplier, **kwargs)

    def minmax_value(self, idx_player : int, **kwargs):
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', self.T) + t0
        player = self.list_players[idx_player]
        def BRD_wrapped(other_action : np.ndarray):
            return player.best_response_over_t(other_action, **kwargs)
        
        def minmax(other_action : np.ndarray):
            other_action = np.reshape(other_action,(self.N-1, tmax),'F')
            player_action = BRD_wrapped(np.sum(other_action, axis =0))
            sum_actions = np.sum(other_action, axis =0) + player_action
            temp = player.scm.evaluate_trajectory(sum_actions, **kwargs)[-1]
            return player.utility_sum_over_t(player_action, sum_actions - player_action, temp = temp, **kwargs)
    
        action_space =  np.delete(self.get_action_space(**kwargs), (idx_player), axis = 0)
        bounds = action_space
        x0 = kwargs.get('x0', action_space[:,t0:tmax,1]/2)
        if x0.shape != bounds[:,t0:tmax,1].flatten().shape:
            x0= x0.flatten('F')
        bounds = Bounds(lb=bounds[:,t0:tmax,0].flatten('F'), ub=bounds[:,t0:tmax,1].flatten('F'), keep_feasible=False)
        res = minimize(minmax, x0 = x0, bounds=bounds,  method='SLSQP', tol=1e-6, options={'maxiter': 10000, 'disp': 0}) 
        if not res.success :
            print(res)
        return res.fun
        

        
    def planning_over_t_piece(self, t_piece, rule, desc, **kwargs):
        a_planning_piece        = np.zeros((self.N,self.T))
        sum_a_planning_piece    = np.zeros(self.T)
        u_planning_piece        = np.zeros((self.N, self.T))
        sum_u_planning_piece    = np.zeros(self.T)
        temp_planning_piece     = np.zeros(self.T)

        for t in tqdm(range(0, self.T, t_piece), desc=desc):
            
            t1=min(t_piece, self.T -t)
            tmax = t_piece
            a, sum_a, u, sum_u, temp = self.planning_game(
            rule, t0= t, tmax=tmax, temperature_target = self.temperature_target, final_multiplier = self.final_multiplier,
            **kwargs)

            a_planning_piece[:,t:t+t_piece] = a[:,:t1]
            sum_a_planning_piece[t:t+t_piece] = sum_a[:t1]
            u_planning_piece[:,t:t+t_piece] = u[:,:t1]
            sum_u_planning_piece[t:t+t_piece] = sum_u[:t1]
            temp_planning_piece[t:t+t_piece] = temp[:t1]

            for emissions in sum_a[:t1] :
                self.scm_game.five_years_cycle_deep(emissions)
        self.reset()
        return a_planning_piece, sum_a_planning_piece, u_planning_piece, sum_u_planning_piece, temp_planning_piece

    def planning_BRD_by_piece(self, t_piece, **kwargs):
        self.ne_a_planning_brd_piece, self.ne_sum_a_planning_brd_piece, self.ne_u_planning_brd_piece, self.ne_sum_u_planning_brd_piece, self.ne_temp_planning_brd_piece = self.planning_over_t_piece(
            t_piece, self.best_response_dynamic_planning, desc= 'Planning BRD, t_piece = {}'.format(t_piece), **kwargs)
        
    def planning_BRD_by_piece_return(self, t_piece, **kwargs):
        return self.planning_over_t_piece(
            t_piece, self.best_response_dynamic_planning, desc= 'Planning BRD, t_piece = {}'.format(t_piece), **kwargs)

    def sum_utilities_planning(self, actions : np.ndarray, **kwargs) -> float:
        tmax = kwargs.get('tmax', self.T)
        actions = np.reshape(actions,(self.N, tmax),'F')
        sum_actions = np.sum(actions, axis =0)
        
        temperature_AT = self.scm_game.evaluate_trajectory(sum_actions, **kwargs)[-1]
        sum_utilities = 0
        for idx, player in enumerate(self.list_players) :
            sum_utilities += player.utility_sum_over_t(actions[idx], sum_actions - actions[idx], temp = temperature_AT, **kwargs )
        return sum_utilities


    def planning_social_optimum_with_constraint(self, **kwargs):
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', self.T) + t0
        constraint_player = kwargs.get('constraint_player', 0)  # Specify the player index for the constraint
        constraint_value = kwargs.get('constraint_value', 0)  # Specify the maximum sum of utilities for the player
        def sum_utilities_planning_wrapped(a):
            return - self.sum_utilities_planning(a, **kwargs)
        
        def constraint_func(a):
            a = np.reshape(a, (self.N, tmax),'F')
            sum_actions = np.sum(a, axis=0)
            temperature_AT = self.scm_game.evaluate_trajectory(sum_actions, **kwargs)[-1]

            player_sum_utility = self.list_players[constraint_player].utility_sum_over_t(
                a[constraint_player], sum_actions - a[constraint_player], temp = temperature_AT, **kwargs
            )
            return player_sum_utility 
        
        action_space = self.get_action_space(**kwargs)
        bounds = action_space
        x0 = kwargs.get('x0', action_space[:, t0:tmax, 1] / 2)
        
        if x0.shape != bounds[:, t0:tmax, 1].flatten().shape:
            x0 = x0.flatten('F')
        
        bounds = Bounds(lb=bounds[:, t0:tmax, 0].flatten('F'), ub=bounds[:, t0:tmax, 1].flatten('F'), keep_feasible=True)
        nlc = NonlinearConstraint(constraint_func, -np.inf, constraint_value)
        
        res = minimize(sum_utilities_planning_wrapped, x0=x0, bounds=bounds, method='SLSQP',
                    tol=1e-6, options={'maxiter': 10000, 'disp': 0}, constraints=[nlc])
        
        if not res.success:
            print(res)
        
        return np.reshape(res.x, (self.N, tmax - t0), 'F')
    

    def planning_social_optimum_with_constraint_temperature(self, **kwargs):
            t0 = kwargs.get('t0', 0)
            tmax = kwargs.get('tmax', self.T) + t0
            constraint_value = kwargs.get('constraint_value', 0)  # Specify the maximum sum of utilities for the player
            def sum_utilities_planning_wrapped(a):
                return - self.sum_utilities_planning(a, **kwargs)
            
            def constraint_func(a):
                a = np.reshape(a, (self.N, tmax),'F')
                sum_actions = np.sum(a, axis=0)
                temperature_AT = self.scm_game.evaluate_trajectory(sum_actions, **kwargs)[-1][-1]

                return temperature_AT - constraint_value
            
            action_space = self.get_action_space(**kwargs)
            bounds = action_space
            x0 = kwargs.get('x0', action_space[:, t0:tmax, 1] / 2)
            
            if x0.shape != bounds[:, t0:tmax, 1].flatten().shape:
                x0 = x0.flatten('F')
            
            bounds = Bounds(lb=bounds[:, t0:tmax, 0].flatten('F'), ub=bounds[:, t0:tmax, 1].flatten('F'), keep_feasible=True)
            nlc = {'type' : 'eq', 'fun' : constraint_func}
            
            res = minimize(sum_utilities_planning_wrapped, x0=x0, bounds=bounds, method='SLSQP',
                        tol=1e-6, options={'maxiter': 10000, 'disp': 0}, constraints=[nlc])
            
            if not res.success:
                print(res)
            
            return np.reshape(res.x, (self.N, tmax - t0), 'F')
        
    # def sum_utilities_planning_weighted(self, actions : np.ndarray, weights :np.ndarray,**kwargs) -> float:
    #     tmax = kwargs.get('tmax', self.T)
    #     actions = np.reshape(actions,(self.N, tmax),'F')
    #     sum_actions = np.sum(actions, axis =0)
        
    #     temperature_AT = self.scm_game.evaluate_trajectory(sum_actions, **kwargs)[-1]
    #     sum_utilities = 0
    #     for idx, player in enumerate(self.list_players) :
    #         sum_utilities += weights[idx] * player.utility_sum_over_t(actions[idx], sum_actions - actions[idx], temp = temperature_AT, **kwargs )
    #     return sum_utilities

    
    # def planning_social_optimum_weighted(self, weights : np.ndarray, **kwargs):
    #     t0 = kwargs.get('t0', 0)
    #     tmax = kwargs.get('tmax', self.T) + t0
    #     def sum_utilities_planning_wrapped(a):
    #         return - self.sum_utilities_planning_weighted(a, weights, **kwargs)
        
    #     action_space = self.get_action_space(**kwargs)
    #     bounds = action_space
    #     x0 = kwargs.get('x0', action_space[:, t0:tmax, 1] / 2)
        
    #     if x0.shape != bounds[:, t0:tmax, 1].flatten().shape:
    #         x0 = x0.flatten('F')
        
    #     bounds = Bounds(lb=bounds[:, t0:tmax, 0].flatten('F'), ub=bounds[:, t0:tmax, 1].flatten('F'), keep_feasible=True)

    #     res = minimize(sum_utilities_planning_wrapped, x0=x0, bounds=bounds, method='SLSQP',
    #                 tol=1e-6, options={'maxiter': 10000, 'disp': 0})
        
    #     if not res.success:
    #         print(res)
        
    #     return np.reshape(res.x, (self.N, tmax - t0), 'F')
    
    def pareto_front(self, nb_points=50, value_constraints = None, **kwargs):
        if self.N >2:
            return 'Err: Number of player superior to 2.'
        
        a_1, a_2 = self.get_action_space(**kwargs)
        if value_constraints is None :
            min1max2 = np.stack((a_1[..., 0], a_2[...,1] ))
            max1min2 = np.stack((a_1[..., 1], a_2[...,0] ))
            val1val2 = self.game_with_strategies_profile(min1max2, np.sum(min1max2, axis=0))[0]
            val2val1 = self.game_with_strategies_profile(max1min2, np.sum(max1min2, axis=0))[0]
            vals = np.stack([val1val2, val2val1])
        else:
            vals = value_constraints
        val = np.stack([np.linspace(*vals[:,0], num=nb_points),np.linspace(*vals[:,1],  num=nb_points)])

        actions = []
        temp = []
        utilities = []
        for idx_ in range(self.N):
            for val_ in tqdm(val[idx_], desc='Player {}'.format(idx_)):
                a_p = self.planning_social_optimum_with_constraint(  constraint_player=idx_, constraint_value=val_, **kwargs)
                actions.append(a_p)
                sum_a_p = np.sum(a_p, axis=0)

                ## Climate modelization 
                # Players' emissions are injected in the SCM.
                temperature_AT =  self.scm_game.evaluate_trajectory(sum_a_p, **kwargs)[-1]
                temp.append(temperature_AT)

                # Calculus of the utilities values for all games.
                u_p = np.zeros(self.N)
                for idx, player in enumerate(self.list_players):
                    u_p[idx] = player.utility_sum_over_t(a_p[idx], sum_a_p - a_p[idx], temp = temperature_AT, **kwargs)
                # Sum of the utilities values for all games.
                utilities.append(u_p)
        actions = np.stack(actions, axis=0)
        utilities = np.array(utilities)
        temp = np.array(temp)
        return actions, utilities, temp

    def pareto_front_multiple(self, indices : tuple[int,int], nb_points=50,**kwargs):
        if len(indices) > 2:
            return 'Err: Number of player superior to 2.'
        i1, i2 = indices
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', self.T) + t0
        a_1, a_2 = self.get_action_space_specific(indices, **kwargs) 
        min1max2 = np.stack((a_1[..., 0], a_2[...,1] ))
        max1min2 = np.stack((a_1[..., 1], a_2[...,0] ))
        val1val2 = self.game_with_strategies_profile(min1max2, np.sum(min1max2, axis=0))[0]
        val2val1 = self.game_with_strategies_profile(max1min2, np.sum(max1min2, axis=0))[0]
        vals = np.stack([val1val2, val2val1])
        val = np.stack([np.linspace(*vals[:,0], num=nb_points),np.linspace(*vals[:,1],  num=nb_points)])
        actions = []
        temp = []
        utilities = []
        for idx_ in range(len(indices)):
            # print(val)
            # print(idx_)
            # print(val[idx_])
            for val_ in tqdm(val[idx_], desc='Player {}'.format(idx_)):
                a_p = self.planning_social_optimum_with_constraint(  constraint_player=idx_, constraint_value=val_, **kwargs)
                actions.append(a_p)
                sum_a_p = np.sum(a_p, axis=0)

                ## Climate modelization 
                # Players' emissions are injected in the SCM.
                temperature_AT =  self.scm_game.evaluate_trajectory(sum_a_p, **kwargs)[-1]
                temp.append(temperature_AT)

                # Calculus of the utilities values for all games.
                u_p = np.zeros(self.N)
                for idx, player in enumerate(self.list_players):
                    u_p[idx] = player.utility_sum_over_t(a_p[idx], sum_a_p - a_p[idx], temp = temperature_AT, **kwargs)
                # Sum of the utilities values for all games.
                utilities.append(u_p)
        actions = np.stack(actions, axis=0)
        utilities = np.array(utilities)
        temp = np.array(temp)
        return actions, utilities, temp

    def planning_inverse_social_optimum_with_constraint(self, **kwargs):
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', self.T) + t0
        constraint_player = kwargs.get('constraint_player', 0)  # Specify the player index for the constraint
        constraint_value = kwargs.get('constraint_value', 0)  # Specify the minimum sum of utilities for the player
        def sum_utilities_planning_wrapped(a):
            return self.sum_utilities_planning(a, **kwargs)
        
        def constraint_func(a):
            a = np.reshape(a, (self.N, tmax),'F')
            sum_actions = np.sum(a, axis=0)
            temperature_AT = self.scm_game.evaluate_trajectory(sum_actions, **kwargs)[-1]

            player_sum_utility = self.list_players[constraint_player].utility_sum_over_t(
                a[constraint_player], sum_actions - a[constraint_player], temp = temperature_AT, **kwargs
            )
            return player_sum_utility 
        
        action_space = self.get_action_space(**kwargs)
        bounds = action_space
        x0 = kwargs.get('x0', action_space[:, t0:tmax, 0])
        
        if x0.shape != bounds[:, t0:tmax, 1].flatten().shape:
            x0 = x0.flatten('F')
        
        bounds = Bounds(lb=bounds[:, t0:tmax, 0].flatten('F'), ub=bounds[:, t0:tmax, 1].flatten('F'), keep_feasible=False)
        nlc = NonlinearConstraint(constraint_func, constraint_value, np.inf)
        constraint = {'type': 'ineq', 'fun': constraint_func}
        
        res = minimize(sum_utilities_planning_wrapped, x0=x0, bounds=bounds, method='SLSQP',
                    tol=1e-6, options={'maxiter': 10000, 'disp': 0}, constraints=[nlc])
        
        if not res.success:
            print(res)
        
        return np.reshape(res.x, (self.N, tmax - t0), 'F')

    def planning_SO_with_constraint(self, **kwargs):
        self.so_a_planning, self.so_sum_a_planning, self.so_u_planning, self.so_sum_u_planning, self.so_temp_planning = self.planning_game(self.planning_social_optimum_with_constraint,
        temperature_target = self.temperature_target, final_multiplier = self.final_multiplier, **kwargs)    
    
    def planning_SO_by_piece_with_constraint(self, t_piece, **kwargs):
        self.so_a_planning_piece, self.so_sum_a_planning_piece, self.so_u_planning_piece, self.so_sum_u_planning_piece, self.so_temp_planning_piece = self.planning_over_t_piece(
            t_piece, self.planning_social_optimum_with_constraint, desc= 'Planning SO, t_piece = {}'.format(t_piece), **kwargs)
        
    def planning_SO_by_piece_with_constraint_return(self, t_piece, **kwargs):
        return self.planning_over_t_piece(
            t_piece, self.planning_social_optimum_with_constraint, desc= 'Planning SO, t_piece = {}'.format(t_piece), **kwargs)

    def planning_SO_inverse_by_piece_with_constraint_return(self, t_piece, **kwargs):
        return self.planning_over_t_piece(
            t_piece, self.planning_inverse_social_optimum_with_constraint, desc= 'Planning SO, t_piece = {}'.format(t_piece), **kwargs)
  
    
    
    def planning_social_optimum(self, **kwargs):
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', self.T) + t0
        def sum_utilities_planning_wrapped(a):
            return - self.sum_utilities_planning(a, **kwargs)
        action_space = self.get_action_space(**kwargs)
        bounds = action_space
        x0 = kwargs.get('x0', action_space[:,t0:tmax,1]/2)
        if x0.shape != bounds[:,t0:tmax,1].flatten().shape:
            x0= x0.flatten('F')
        bounds = Bounds(lb=bounds[:,t0:tmax,0].flatten('F'), ub=bounds[:,t0:tmax,1].flatten('F'), keep_feasible=True)
        # start_time = time.time()

        # res = minimize(sum_utilities_planning_wrapped ,x0=x0, method='SLSQP',  tol=1e-6 , bounds=bounds, options={'maxiter': 1000, 'disp': 0})
        # end_time = time.time()
        # print('SO SLSQP', end_time - start_time)
        # start_time1 = time.time()

        res = minimize(sum_utilities_planning_wrapped, x0 = x0, bounds=bounds,  method='L-BFGS-B', tol=1e-6, options={'maxiter': 10000, 'disp': 0}) #, constraints=constraint)#jac=jac,, constraints=constraint)
        # res = minimize(response, x0 = x0, bounds=bounds,  method='SLSQP', tol=1e-6, options={'maxiter': 10000, 'disp': 0}) #, constraints=constraint)#jac=jac,, constraints=constraint)
        # res = minimize(response, x0 = x0, bounds=bounds,  method='SLSQP', tol=1e-6, options={'maxiter': 10000, 'disp': 0}) #, constraints=constraint)#jac=jac,, constraints=constraint)
        # end_time1 = time.time()
        # print('SO L-BFGS-B:', end_time1 - start_time1)
        return np.reshape(res.x  ,(self.N, tmax-t0),'F')
    
    def planning_SO(self, **kwargs):
        self.so_a_planning, self.so_sum_a_planning, self.so_u_planning, self.so_sum_u_planning, self.so_temp_planning = self.planning_game(self.planning_social_optimum,
        temperature_target = self.temperature_target, final_multiplier = self.final_multiplier, **kwargs)    
    
    def planning_SO_by_piece(self, t_piece, **kwargs):
        self.so_a_planning_piece, self.so_sum_a_planning_piece, self.so_u_planning_piece, self.so_sum_u_planning_piece, self.so_temp_planning_piece = self.planning_over_t_piece(
            t_piece, self.planning_social_optimum, desc= 'Planning SO, t_piece = {}'.format(t_piece), **kwargs)
        
    def planning_SO_by_piece_return(self, t_piece, **kwargs):
        return self.planning_over_t_piece(
            t_piece, self.planning_social_optimum, desc= 'Planning SO, t_piece = {}'.format(t_piece), **kwargs)
    
    def repeated_one_shot_game_NE(self, **kwargs)-> None:
        kwargs.pop('t_piece', None)
        self.ne_a_p, self.ne_sum_a_p, self.ne_u_p, self.ne_sum_u_p, self.ne_temp_p = self.planning_over_t_piece(
            1, self.best_response_dynamic_planning, desc= 'Repeated One-shot BRD',**kwargs)

    def repeated_one_shot_game_SO(self, **kwargs)-> None:
        kwargs.pop('t_piece', None)
        self.so_a_p, self.so_sum_a_p, self.so_u_p, self.so_sum_u_p, self.so_temp_p = self.planning_over_t_piece(
            1, self.planning_social_optimum, desc= 'Repeated One-shot SO',**kwargs)

    def receding_over_t_piece(self, t_piece, rule, desc, **kwargs):
            a_planning_piece        = np.zeros((self.N,self.T))
            sum_a_planning_piece    = np.zeros(self.T)
            u_planning_piece        = np.zeros((self.N, self.T))
            sum_u_planning_piece    = np.zeros(self.T)
            temp_planning_piece     = np.zeros(self.T)

            
            for t in tqdm(range(0, self.T, 1), desc=desc):
                t1=min(t_piece, self.T -t)
                tmax = t_piece
                a, sum_a, u, sum_u, temp = self.planning_game(
                rule, t0= t, tmax=tmax, temperature_target = self.temperature_target, final_multiplier = self.final_multiplier,
                **kwargs)

                a_planning_piece[:,t:t+t_piece] = a[:,:t1]
                sum_a_planning_piece[t:t+t_piece] = sum_a[:t1]
                u_planning_piece[:,t:t+t_piece] = u[:,:t1]
                sum_u_planning_piece[t:t+t_piece] = sum_u[:t1]
                temp_planning_piece[t:t+t_piece] = temp[:t1]


                self.scm_game.five_years_cycle_deep(sum_a[0])
            self.reset()
            return a_planning_piece, sum_a_planning_piece, u_planning_piece, sum_u_planning_piece, temp_planning_piece
    
    def receding_BRD_by_piece(self, t_piece, **kwargs):
        self.ne_a_planning_brd_piece, self.ne_sum_a_planning_brd_piece, self.ne_u_planning_brd_piece, self.ne_sum_u_planning_brd_piece, self.ne_temp_planning_brd_piece = self.receding_over_t_piece(
            t_piece, self.best_response_dynamic_planning, desc= 'Receding BRD, t_piece = {}'.format(t_piece), **kwargs)
        

    def receding_SO_by_piece(self, t_piece, **kwargs):
        self.so_a_planning_piece, self.so_sum_a_planning_piece, self.so_u_planning_piece, self.so_sum_u_planning_piece, self.so_temp_planning_piece = self.receding_over_t_piece(
            t_piece, self.planning_social_optimum, desc= 'Receding SO, t_piece = {}'.format(t_piece), **kwargs)
        
    def receding_BRD_by_piece_return(self, t_piece, **kwargs):
        return self.receding_over_t_piece(
            t_piece, self.best_response_dynamic_planning, desc= 'Receding BRD, t_piece = {}'.format(t_piece), **kwargs)
        

    def receding_SO_by_piece_return(self, t_piece, **kwargs):
        return self.receding_over_t_piece(
            t_piece, self.planning_social_optimum, desc= 'Receding SO, t_piece = {}'.format(t_piece), **kwargs)

    def planning_BRD_by_piece_(self, t_piece, **kwargs):
        self.ne_a_planning_brd_piece        = np.zeros((self.N,self.T))
        self.ne_sum_a_planning_brd_piece    = np.zeros(self.T)
        self.ne_u_planning_brd_piece        = np.zeros((self.N, self.T))
        self.ne_sum_u_planning_brd_piece    = np.zeros(self.T)
        self.ne_temp_planning_brd_piece     = np.zeros(self.T)

        for t in range(self.T):

            tmax=min(t_piece, self.T -t)
            ne_a_planning_brd_piece, ne_sum_a_planning_brd_piece, ne_u_planning_brd_piece, ne_sum_u_planning_brd_piece, ne_temp_planning_brd_piece = self.planning_game(
            self.best_response_dynamic_planning, t0= t, tmax=tmax, temperature_target = self.temperature_target, final_multiplier = self.final_multiplier,
            **kwargs)

            self.ne_a_planning_brd_piece[:,t:] = ne_a_planning_brd_piece
            self.ne_sum_a_planning_brd_piece[t:] = ne_sum_a_planning_brd_piece
            self.ne_u_planning_brd_piece[:,t:] = ne_u_planning_brd_piece
            self.ne_sum_u_planning_brd_piece[t:] = ne_sum_u_planning_brd_piece
            self.ne_temp_planning_brd_piece[t:] = ne_temp_planning_brd_piece

            self.scm_game.five_years_cycle_deep(ne_sum_a_planning_brd_piece[0])


    
    def generate_efficiency_table(self, functions, repetitions=10, **kwargs):
        efficiency_table = []

        for func_name in functions:
            func = getattr(self, func_name, None)
            if func is None:
                print(f"Function '{func_name}' not found in the class.")
                continue

            print(f"Running '{func_name}' for {repetitions} repetitions...")
            start_time = time.time()
            results = []

            for _ in range(repetitions):
                # Run the function and measure the time taken
                func_start_time = time.time()
                func_result = func(**kwargs)
                func_end_time = time.time()
                func_time = func_end_time - func_start_time

                # Append the result and time to the results list
                results.append(func_result)

            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / repetitions

            # Add the function name, average time, and other metrics to the table
            efficiency_table.append([func_name, avg_time, len(results)])

        # Create the table with tabulate
        table_headers = ["Function", "Avg Time (s)", "Repetitions"]
        efficiency_table_str = tabulate(efficiency_table, headers=table_headers, tablefmt="latex")

        return efficiency_table_str


    def jacobian_sum_benefice(self, a : np.ndarray):
        tmax = a.shape[1]
        diag = [derivative(partial(player.benefit, t = t), a[idx,t],  order=7) for t in range(tmax) for idx, player in enumerate(self.list_players)]
        return np.array(diag)

    def jacobian_sum_damage_planning(self, a : np.ndarray, **kwargs):
        tmax = a.shape[1]

        sum_a = np.sum(a, axis=0)

        CC = self.scm_game.carbon_model
        TD = self.scm_game.temperature_model

        carbon_AT, forcing, temperature_AT = self.scm_game.evaluate_trajectory(sum_a, tmax=tmax, **kwargs)

        jac_carbon_AT = jacobian_linear_model(CC.Ac, CC.bc, CC.dc, tmax)
        jac_forcing = jacobian_forcing(carbon_AT)
        jac_temperature_AT = jacobian_linear_model(TD.At, TD.bt, TD.dt, tmax)
        jac_damage = sum(self.deltas) * jacobian_damage_function(temperature_AT, self.list_damage_function[0])
        jac_sum = jacobian_sum(tmax, self.N)
        print(jac_damage.shape, jac_temperature_AT.shape, jac_forcing.shape, jac_carbon_AT.shape, jac_sum.shape, )
        # pas le bon jacobien car il y a un problem sur les fonction dammages
        jacobian =  jac_damage @ jac_temperature_AT @ jac_forcing @ jac_carbon_AT @ jac_sum 

        return jacobian

    def jacobian_sum_utilities_planning(self, a : np.ndarray, **kwargs):
        tmax = kwargs.get('tmax', self.T)

        a = np.reshape(a,(self.N, tmax),'F') # the reshape function is weird it's important
        return self.jacobian_sum_benefice(a) - self.jacobian_sum_damage_planning(a, **kwargs)


    def prod_utilities(self, actions : np.ndarray, profile : np.ndarray, **kwargs):
        tmax = kwargs.get('tmax', self.T)
        actions = np.reshape(actions,(self.N, tmax),'F')
        sum_actions = np.sum(actions, axis =0)


        temperature_AT = self.scm_game.evaluate_trajectory(sum_actions, **kwargs)[-1]
        prod_utilities = 1
        for idx, player in enumerate(self.list_players) :
            prod_utilities *= player.utility_sum_over_t(actions[idx], sum_actions - actions[idx], temp = temperature_AT, **kwargs ) - profile[idx] 
        return prod_utilities
    
    def utilities_minus_profile(self, actions : np.ndarray, profile : np.ndarray,  **kwargs):
        tmax = kwargs.get('tmax', self.T)
        actions = np.reshape(actions,(self.N, tmax),'F')
        sum_actions = np.sum(actions, axis =0)
        utilities = np.zeros((self.N))

        temperature_AT = self.scm_game.evaluate_trajectory(sum_actions, **kwargs)[-1]
        for idx, player in enumerate(self.list_players) :
            utilities[idx] = player.utility_sum_over_t(actions[idx], sum_actions - actions[idx], temp = temperature_AT, **kwargs ) - profile[idx]
        return utilities

    def nash_barganing_solution(self, utility_profile : np.ndarray, **kwargs):
        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', self.T) + t0
        def prod_utilities(a):
            return - self.prod_utilities(a, utility_profile, **kwargs)
        def utilities_minus_profile(a):
            return  self.utilities_minus_profile(a, utility_profile, **kwargs)
        action_space = self.get_action_space(**kwargs)
        bounds = action_space
        x0 = kwargs.get('x0', action_space[:,t0:tmax,1]/2)
        if x0.shape != bounds[:,t0:tmax,1].flatten().shape:
            x0= x0.flatten('F')
        bounds = Bounds(lb=bounds[:,t0:tmax,0].flatten('F'), ub=bounds[:,t0:tmax,1].flatten('F'))
        constraint = NonlinearConstraint(utilities_minus_profile, ub = np.ones_like(utility_profile) * 1e9, lb = - np.zeros_like(utility_profile) * 1e-9)
        res = minimize(prod_utilities ,x0=x0, constraints=constraint,  tol=1e-6 , bounds=bounds, options={'maxiter': 1000, 'disp': 1})
        return np.reshape(res.x  ,(self.N, tmax-t0),'F'), res



    def potential_planning(self, actions : np.ndarray, **kwargs):

        tmax = kwargs.get('tmax', self.T)
        if not actions.shape == (self.N, tmax):
            actions = np.reshape(actions,(self.N, tmax),'F')
        sum_actions = np.sum(actions, axis =0)
        
        potential = -self.list_players[0].damages(sum_actions, **kwargs) / self.list_players[0].delta
        for idx, player in enumerate(self.list_players) :
            potential += player.benefit(actions[idx], **kwargs) /player.delta 
        
        return potential
    
    def gradient_over_potential(self, **kwargs):

        t0 = kwargs.get('t0', 0)
        tmax = kwargs.get('tmax', self.T) + t0
        action_space = self.get_action_space(**kwargs)
        def potential_planning_wrapped(a):
            return - self.potential_planning(a, **kwargs)

        bounds = action_space

        x0 = kwargs.get('x0', action_space[:,t0:tmax,1]/2)
        if x0.shape != bounds[:,t0:tmax,1].flatten().shape:
            x0= x0.flatten('F')
        bounds = Bounds(lb=bounds[:,t0:tmax,0].flatten('F'), ub=bounds[:,t0:tmax,1].flatten('F'), keep_feasible=True)
        res = minimize(potential_planning_wrapped ,x0=x0 , method='SLSQP', bounds=bounds, tol = 1e-6, options={'maxiter': 1000, 'disp': 0})
        return np.reshape(res.x  ,(self.N, tmax),'F')   



    # def planning_social_optimum(self, **kwargs):
    #     t0 = kwargs.get('t0', 0)
    #     tmax = kwargs.get('tmax', self.T) + t0
    #     def sum_utilities_planning_wrapped(a):
    #         return - self.sum_utilities_planning(a, **kwargs)
    #     action_space = self.get_action_space(**kwargs)
    #     bounds = action_space
    #     x0 = kwargs.get('x0', action_space[:,t0:tmax,1]/2)
    #     if x0.shape != bounds[:,t0:tmax,1].flatten().shape:
    #         x0= x0.flatten('F')
    #     bounds = Bounds(lb=bounds[:,t0:tmax,0].flatten('F'), ub=bounds[:,t0:tmax,1].flatten('F'), keep_feasible=True)
    #     start_time = time.time()

    #     res = minimize(sum_utilities_planning_wrapped ,x0=x0, method='SLSQP',  tol=1e-6 , bounds=bounds, options={'maxiter': 1000, 'disp': 0})
    #     end_time = time.time()
    #     print('SO SLSQP', end_time - start_time)
    #     start_time1 = time.time()

    #     res = minimize(sum_utilities_planning_wrapped, x0 = x0, bounds=bounds,  method='L-BFGS-B', tol=1e-6, options={'maxiter': 10000, 'disp': 0}) #, constraints=constraint)#jac=jac,, constraints=constraint)
    #     # res = minimize(response, x0 = x0, bounds=bounds,  method='SLSQP', tol=1e-6, options={'maxiter': 10000, 'disp': 0}) #, constraints=constraint)#jac=jac,, constraints=constraint)
    #     # res = minimize(response, x0 = x0, bounds=bounds,  method='SLSQP', tol=1e-6, options={'maxiter': 10000, 'disp': 0}) #, constraints=constraint)#jac=jac,, constraints=constraint)
    #     end_time1 = time.time()
    #     print('SO L-BFGS-B:', end_time1 - start_time1)
    #     return np.reshape(res.x  ,(self.N, tmax-t0),'F')

    # def jacobian_damage_planning_all_players(self, a : np.ndarray, **kwargs):
    #     tmax = a.shape[1]

    #     sum_a = np.sum(a, axis=0)

    #     CC = self.scm_game.carbon_model
    #     TD = self.scm_game.temperature_model

    #     carbon_AT, forcing, temperature_AT = self.scm_game.evaluate_trajectory(sum_a, tmax=tmax, **kwargs)

    #     jac_carbon_AT = jacobian_linear_model(CC.Ac, CC.bc, CC.dc, tmax)
    #     jac_forcing = jacobian_forcing(carbon_AT)
    #     jac_temperature_AT = jacobian_linear_model(TD.At, TD.bt, TD.dt, tmax)
    #     jac_damage = jacobian_damage_function(temperature_AT, self.damage_function)
    #     jac_sum = jacobian_sum(tmax, self.N)
        
    #     jacobian =  jac_damage @ jac_temperature_AT @ jac_forcing @ jac_carbon_AT @ jac_sum

    #     return jacobian


    # def jacobian_benefice(self, a : np.ndarray):
    #     tmax = a.shape[1]
    #     diag = [1/player.delta *  derivative(player.benefit_function ,a[idx,t]) for t in range(tmax) for idx, player in enumerate(self.list_players)]
    #     return np.array(diag).T

    # def jacobian_potential_planning(self, tmax,  a : np.ndarray, **kwargs):
    #     a = np.reshape(a,(self.N, tmax),'F') # the reshape function is weird it's important
    #     return self.jacobian_benefice(a) - self.jacobian_damage_planning_all_players(a, **kwargs)

    # def gradient_over_potential(self, tmax, **kwargs):
    #     def potential_planning_wrapped(a):
    #         return - self.potential_planning(tmax, a, **kwargs)
    #     def jacobian_potential_planning_wrapped(a):
    #         return - self.jacobian_potential_planning(tmax, a, **kwargs)

    #     action_space = self.get_action_space()
    #     bounds = bounds = action_space[:,1,:]

    #     x0 = kwargs.get('x0', bounds[:,1])
    #     if x0.shape != bounds[:,1].shape:
    #         x0= x0.flatten()
    #         print('pass')
    #     bounds = Bounds(lb=bounds[:,0], ub=bounds[:,1], keep_feasible=True)
    #     # print(potential_planning_wrapped(x0).shape)
    #     res = minimize(potential_planning_wrapped ,x0=x0 , bounds=bounds, jac = jacobian_potential_planning_wrapped, tol = 1e-6, options={'maxiter': 1000, 'disp': 0})
    #     return np.reshape(res.x  ,(self.N, tmax),'F')   

    # def planning_SO_by_piece_(self, t_piece, **kwargs):
    #     self.so_a_planning_piece        = np.zeros((self.N,self.T))
    #     self.so_sum_a_planning_piece    = np.zeros(self.T)
    #     self.so_u_planning_piece        = np.zeros((self.N, self.T))
    #     self.so_sum_u_planning_piece    = np.zeros(self.T)
    #     self.so_temp_planning_piece     = np.zeros(self.T)

    #     # carbon_state = self.scm_game.carbon_state
    #     # temperature_state = self.scm_game.temperature_state
        
    #     for t in range(self.T):
    #         # print(carbon_state)
    #         # print(temperature_state)
    #         tmax=min(t_piece, self.T -t)

    #         a, sum_a, u, sum_u, temp = self.planning_game(
    #         self.planning_social_optimum, t0= t, tmax=tmax, temperature_target = self.temperature_target, final_multiplier = self.final_multiplier,
    #         **kwargs)

    #         self.so_a_planning_piece[:,t] = a[:,0]
    #         self.so_sum_a_planning_piece[t] = sum_a[0]
    #         self.so_u_planning_piece[:,t] = u[:,0]
    #         self.so_sum_u_planning_piece[t] = sum_u[0]
    #         self.so_temp_planning_piece[t] = temp[0]

    #         self.scm_game.five_years_cycle_deep(sum_a[0])
    #         # carbon_state = self.scm_game.carbon_state
    #         # temperature_state = self.scm_game.temperature_state




    # def planning_game_with_strategies_profile(self, array_action : np.ndarray, sum_action: np.ndarray, **kwargs) -> None:
    #     utilities = np.zeros(self.N,)

    #     for indice in range(self.N):
    #         player = self.list_players[indice]
    #         utilities[indice] = player.utility_sum_over_t(array_action[indice],
    #                                                                  sum_action - array_action[indice], **kwargs)
                                                                     

    #     return utilities
    
    # def planning_game_pareto_front(self, action_1 : np.ndarray) -> None:
        # tmax = action_1.shape[0]

        # def sum_utilities_planning(tmax, actions : np.ndarray, **kwargs) -> float:
        #     sum_actions = actions + action_1
        #     temperature_AT = self.scm_game.evaluate_trajectory(sum_actions, **kwargs)[-1]
        #     sum_utilities = 0
        #     player = self.list_players[1]
        #     sum_utilities += np.sum(player.utility_sum_over_t(actions, action_1 , temp = temperature_AT, **kwargs ))
        #     player = self.list_players[0]

        #     sum_utilities += np.sum(player.utility_sum_over_t(action_1, actions , temp = temperature_AT, **kwargs ))


        #     return sum_utilities

        # def sum_utilities_planning_wrapped(a):
        #     return - sum_utilities_planning(tmax, a)


        # action_space = self.get_action_space()
        # bounds = np.tile(action_space[1], (tmax,1))
        # x0 = bounds[:,1]
        # bounds = Bounds(lb=bounds[:,0], ub=bounds[:,1], keep_feasible=True)

        # # res = minimize(sum_utilities_planning_wrapped ,x0=x0 , bounds=bounds, jac = jacobian_sum_utilities_planning_wrapped, tol = 1e-6, options={'maxiter': 1000, 'disp': 0})
        # res = minimize(sum_utilities_planning_wrapped ,x0=x0 , bounds=bounds, options={'maxiter': 1000, 'disp': 0})
        # action_2 = res.x 
    
        # utilities = np.zeros(self.N)

        # # temp = self.scm_game.five_years_atmospheric_temp(sum_action)
        # player = self.list_players[1]
        # utilities[1] = player.utility_sum_over_t(action_2, action_1)
        # player = self.list_players[0]
        # utilities[0] = player.utility_sum_over_t(action_1,action_2)                                                           

        # return utilities