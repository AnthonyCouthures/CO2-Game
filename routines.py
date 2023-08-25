
from pathlib import Path
location = 'ca'

path = Path(location)

from joblib import Memory
memory = Memory(path, verbose=1)

from parameters import *
from models.game_theory_model import *
from models.game import *


@memory.cache
def classic_game(list_param, list_label, horizon):
    
    print('Running f(%s)' )

    # initialize empty lists to store results
    list_ne = []
    list_so = []
    list_ne_u = []
    list_so_u = []
    list_ne_planning_u = []
    list_so_planning_u = []
    list_sum_action = []
    list_sum_utilities = []
    list_temp = []
    list_sum_action_so = []
    list_sum_utilities_so = []
    list_temp_so = []
    list_sum_action_planning = []
    list_sum_utilities_planning = []
    list_temp_planning = []
    list_sum_action_planning_so = []
    list_sum_utilities_planning_so = []
    list_temp_planning_so = []
    list_ne_planning = []
    list_so_planning = []

    # loop over parameter and label lists
    for param, label in zip(list_param, list_label):
        damage, alpha, list_benefit_functions = param
        print('coef:', label[0], 'alpha:', label[1], 'function:', label[2][0])
        
        # create list of players
        list_players = create_players(alpha=alpha, list_benefit_functions=list_benefit_functions, damage_function=damage, discount=1)

        # create game instance
        game = Game(list_players, horizon=horizon)

        # run repeated one-shot game for Nash equilibria
        game.repeated_one_shot_game_NE()
        game.reset()

        # run repeated one-shot game for social optimum
        game.repeated_one_shot_game_SO()
        game.reset()

        # run planning for best response dynamics
        game.planning_BRD()
        game.planning_SO()

        # store results in respective lists
        list_ne.append(game.ne_a_p)
        list_ne_u.append(game.ne_u_p)
        list_ne_planning_u.append(game.ne_u_planning_brd)
        
        list_so.append(game.so_a_p)
        list_so_u.append(game.so_u_p)
        list_so_planning_u.append(game.so_u_planning)
        
        list_sum_action.append(game.ne_sum_a_p)
        list_temp.append(game.ne_temp_p)
        list_sum_utilities.append(game.ne_sum_u_p)
        
        list_sum_action_so.append(game.so_sum_a_p)
        list_temp_so.append(game.so_temp_p)
        list_sum_utilities_so.append(game.so_sum_u_p)

        list_ne_planning.append(game.ne_a_planning_brd)
        list_sum_action_planning.append(game.ne_sum_a_planning_brd)
        list_temp_planning.append(game.ne_temp_planning_brd)
        list_sum_utilities_planning.append(game.ne_sum_u_planning_brd)

        list_so_planning.append(game.so_a_planning)
        list_sum_action_planning_so.append(game.so_sum_a_planning)
        list_temp_planning_so.append(game.so_temp_planning)
        list_sum_utilities_planning_so.append(game.so_sum_u_planning)

        print('-----------------------------------------------------------')


    return (list_ne, list_so, list_ne_u, list_so_u, list_ne_planning_u, list_so_planning_u, 
            list_sum_action, list_sum_utilities, list_temp, list_sum_action_so, list_sum_utilities_so, list_temp_so, 
            list_sum_action_planning, list_sum_utilities_planning, list_temp_planning, list_sum_action_planning_so, 
            list_sum_utilities_planning_so, list_temp_planning_so, list_ne_planning, list_so_planning)


@memory.cache
def comparing_game_old(list_param,  list_label, list_t_piece, horizon):

    list_list_sum_action_planning = []
    list_list_sum_utilities_planning = []
    list_list_temp_planning = []
    list_list_ne_planning = []
    list_list_ne_planning_u = []
    list_list_sum_action_planning_so = []
    list_list_sum_utilities_planning_so = []
    list_list_temp_planning_so = []
    list_list_so_planning = []
    list_list_so_planning_u = []

    for param, label in zip(list_param,list_label):
        damage, alpha, list_benefit_functions = param
        print('coef :', label[0], 'alpha :', label[1], 'function :', label[2][0])

        list_sum_action_planning = []
        list_sum_utilities_planning = []
        list_temp_planning = []
        list_ne_planning = []
        list_ne_planning_u = []

        list_sum_action_planning_so = []
        list_sum_utilities_planning_so = []
        list_temp_planning_so = []
        list_so_planning = []
        list_so_planning_u = []
        list_result_ne = []
        list_result_so = []

        for t_piece in list_t_piece :
            list_players = create_players(alpha=alpha, list_benefit_functions=list_benefit_functions,  damage_function=damage, discount=1)

            game = Game(list_players, horizon=horizon)

            list_result_ne.append(game.planning_BRD_by_piece_return(t_piece=t_piece))
            game.reset()
            list_result_so.append(game.planning_SO_by_piece_return(t_piece=t_piece))
            game.reset()

        list_ne_planning, list_sum_action_planning, list_temp_planning, list_ne_planning_u, list_sum_utilities_planning = zip(*list_result_ne)
        list_so_planning, list_sum_action_planning_so, list_temp_planning_so, list_so_planning_u, list_sum_utilities_planning_so = zip(*list_result_so)


        list_list_ne_planning.append(list_ne_planning)
        list_list_sum_action_planning.append(list_sum_action_planning)
        list_list_temp_planning.append(list_temp_planning)
        list_list_ne_planning_u.append(list_ne_planning_u)
        list_list_sum_utilities_planning.append(list_sum_utilities_planning)
        list_list_so_planning.append(list_so_planning)
        list_list_sum_action_planning_so.append(list_sum_action_planning_so)
        list_list_temp_planning_so.append(list_temp_planning_so)
        list_list_so_planning_u.append(list_so_planning_u)
        list_list_sum_utilities_planning_so.append(list_sum_utilities_planning_so)
        print('----------------------------------------')

    return (list_list_sum_action_planning, list_list_sum_utilities_planning, list_list_temp_planning, 
            list_list_ne_planning, list_list_ne_planning_u, list_list_sum_action_planning_so, 
            list_list_sum_utilities_planning_so, list_list_temp_planning_so, list_list_so_planning, 
            list_list_so_planning_u)


@memory.cache
def planning_game_BRD_t_piece(param, t_piece, horizon):

    damage, alpha, list_benefit_functions = param

    list_players = create_players(alpha=alpha, list_benefit_functions=list_benefit_functions, damage_function=damage, discount=1)

    game = Game(list_players, horizon=horizon)

    return game.planning_BRD_by_piece_return(t_piece=t_piece)

@memory.cache
def planning_game_SO_t_piece(param, t_piece, horizon):

    damage, alpha, list_benefit_functions = param

    list_players = create_players(alpha=alpha, list_benefit_functions=list_benefit_functions, damage_function=damage, discount=1)

    game = Game(list_players, horizon=horizon)

    return game.planning_SO_by_piece_return(t_piece=t_piece)


@memory.cache
def planning_game(param, label, horizon, list_t_piece):
    print('coef :', label[0], 'alpha :', label[1], 'function :', label[2][0])

    result_ne = []
    result_so = []

    for t_piece in list_t_piece:
        result_ne.append(planning_game_BRD_t_piece(param=param, t_piece=t_piece, horizon=horizon))
        result_so.append(planning_game_SO_t_piece(param=param, t_piece=t_piece, horizon=horizon))
    list_ne_planning, list_sum_action_planning, list_ne_planning_u, list_sum_utilities_planning, list_temp_planning = zip(*result_ne)
    list_so_planning, list_sum_action_planning_so, list_so_planning_u, list_sum_utilities_planning_so, list_temp_planning_so = zip(*result_so)
    print('----------------------------------------')

    return (list_sum_action_planning, list_sum_utilities_planning, list_ne_planning, list_ne_planning_u, list_temp_planning, list_sum_action_planning_so, list_sum_utilities_planning_so, list_so_planning, list_so_planning_u, list_temp_planning_so)



@memory.cache
def planning_game_old(param, label, horizon, list_t_piece):
    damage, alpha, list_benefit_functions = param
    print('coef :', label[0], 'alpha :', label[1], 'function :', label[2][0])

    result_ne = []
    result_so = []

    for t_piece in list_t_piece:
        list_players = create_players(alpha=alpha, list_benefit_functions=list_benefit_functions, damage_function=damage, discount=1)

        game = Game(list_players, horizon=horizon)

        result_ne.append(game.planning_BRD_by_piece_return(t_piece=t_piece))
        game.reset()
        result_so.append(game.planning_SO_by_piece_return(t_piece=t_piece))
    list_ne_planning, list_sum_action_planning, list_ne_planning_u, list_sum_utilities_planning, list_temp_planning = zip(*result_ne)
    list_so_planning, list_sum_action_planning_so, list_so_planning_u, list_sum_utilities_planning_so, list_temp_planning_so = zip(*result_so)
    print('----------------------------------------')

    return (list_sum_action_planning, list_sum_utilities_planning, list_ne_planning, list_ne_planning_u, list_temp_planning, list_sum_action_planning_so, list_sum_utilities_planning_so, list_so_planning, list_so_planning_u, list_temp_planning_so)

@memory.cache
def comparing_planning_game_old(list_param, list_label, list_t_piece, horizon):
    result = []

    for param, label in zip(list_param,list_label):
        damage, alpha, list_benefit_functions = param
        print('coef :', label[0], 'alpha :', label[1], 'function :', label[2][0])

        result_ne = []
        result_so = []

        for t_piece in list_t_piece:
            list_players = create_players(alpha=alpha, list_benefit_functions=list_benefit_functions, damage_function=damage, discount=1)

            game = Game(list_players, horizon=horizon)

            result_ne.append(game.planning_BRD_by_piece_return(t_piece=t_piece))
            game.reset()
            result_so.append(game.planning_SO_by_piece_return(t_piece=t_piece))
        list_ne_planning, list_sum_action_planning, list_ne_planning_u, list_sum_utilities_planning, list_temp_planning = zip(*result_ne)
        list_so_planning, list_sum_action_planning_so, list_so_planning_u, list_sum_utilities_planning_so, list_temp_planning_so = zip(*result_so)

        result.append((list_sum_action_planning, list_sum_utilities_planning, list_ne_planning, list_ne_planning_u, list_temp_planning, list_sum_action_planning_so, list_sum_utilities_planning_so, list_so_planning, list_so_planning_u, list_temp_planning_so))
        print('----------------------------------------')

    return list(zip(*result))


@memory.cache
def comparing_planning_game(list_param, list_label, list_t_piece, horizon):
    result = []

    for param, label in zip(list_param,list_label):

        result.append(planning_game(param, label, horizon, list_t_piece))

    return list(zip(*result))


@memory.cache
def receding_game_BRD_t_piece(param, t_piece, horizon):

    damage, alpha, list_benefit_functions = param

    list_players = create_players(alpha=alpha, list_benefit_functions=list_benefit_functions, damage_function=damage, discount=1)

    game = Game(list_players, horizon=horizon)

    return game.receding_BRD_by_piece_return(t_piece=t_piece)

@memory.cache
def receding_game_SO_t_piece(param, t_piece, horizon):

    damage, alpha, list_benefit_functions = param

    list_players = create_players(alpha=alpha, list_benefit_functions=list_benefit_functions, damage_function=damage, discount=1)

    game = Game(list_players, horizon=horizon)

    return game.receding_SO_by_piece_return(t_piece=t_piece)



@memory.cache
def receding_game(param, label, horizon, list_t_piece):
    print('coef :', label[0], 'alpha :', label[1], 'function :', label[2][0])

    result_ne = []
    result_so = []

    for t_piece in list_t_piece:
        result_ne.append(receding_game_BRD_t_piece(param=param, t_piece=t_piece, horizon=horizon))
        result_so.append(receding_game_SO_t_piece(param=param, t_piece=t_piece, horizon=horizon))
    list_ne_planning, list_sum_action_planning, list_ne_planning_u, list_sum_utilities_planning, list_temp_planning = zip(*result_ne)
    list_so_planning, list_sum_action_planning_so, list_so_planning_u, list_sum_utilities_planning_so, list_temp_planning_so = zip(*result_so)
    print('----------------------------------------')

    return (list_sum_action_planning, list_sum_utilities_planning, list_ne_planning, list_ne_planning_u, list_temp_planning, list_sum_action_planning_so, list_sum_utilities_planning_so, list_so_planning, list_so_planning_u, list_temp_planning_so)


@memory.cache
def comparing_receding_game(list_param, list_label, list_t_piece, horizon):
    result = []

    for param, label in zip(list_param,list_label):
        result.append(receding_game(param, label, horizon, list_t_piece))


    return list(zip(*result))

