from turtle import title
import matplotlib.pyplot as plt
import numpy as np
from parameters import *
from models.geophysic_models import *
import seaborn as sns
from models.game import *

def plot_one_atmospheric_temperature(array_emission : np.ndarray,
                                    carbon_model : Linear_Carbon_Model,
                                    temp_model : Linear_Temperature_Dynamic,
                                    title : str = None) :

    c_name = carbon_model.name
    t_name = temp_model.name

    IAM = Simple_Climate_Model(carbon_model=carbon_model, temperature_model=temp_model)
    array_carbon, array_forcing, array_temperature, array_atmospheric_temp = IAM.multiple_cycles(T,
                                    array_emission, E_EX, F_EX)

    x_axis = np.arange(FIRST_YEAR, FINAL_YEAR + 5, 5)

    plt.figure(figsize=(15, 9), dpi=100,  tight_layout=True)
    sns.set_style('darkgrid',  {"grid.color": ".6", "grid.linestyle": ":"})
    sns.set_context("paper")
    # plt.grid(which='both')

    plt.plot(x_axis, array_atmospheric_temp, label='{}-{}'.format(c_name, t_name))
    plt.title(title)
    plt.ylabel('Global mean surface temperature \n of the atmosphere', multialignment='center')

    plt.legend()
    plt.show()

def plot_list_atmospheric_temperature(array_emission : np.ndarray,
                                        list_carbon_model : list[Linear_Carbon_Model],
                                        list_temp_model : list[Linear_Temperature_Dynamic],
                                        title : str = None,
                                        save : bool = False) :
    plt.figure(figsize=(8, 4), dpi=100, tight_layout=True)
    # plt.grid(which='both')
    sns.set_style('darkgrid',  {"grid.color": ".6", "grid.linestyle": ":"})
    sns.set_context("paper")
    for temp_model in list_temp_model:
        t_name = temp_model.name
        for carbon_model in list_carbon_model:
            temp_model.reset()
            temp_model.init_temp_dynamic(np.array([1.01, 0.0068]))
            c_name = carbon_model.name

            IAM = Simple_Climate_Model(carbon_model=carbon_model, temperature_model=temp_model)
            array_carbon, array_forcing, array_temperature, array_atmospheric_temp = IAM.multiple_cycles(T,
                                            array_emission, E_EX, F_EX)

            x_axis = np.arange(FIRST_YEAR, FINAL_YEAR + 5, 5)

            

            plt.plot(x_axis, array_atmospheric_temp, '--', label='{}'.format(c_name))
    plt.title('{} with {}'.format(title,t_name), fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('Global mean surface temperature \n of the atmosphere in °C', multialignment='center', fontsize=13)


    plt.legend(prop={'size': 13})
    if save:
        plt.savefig('plots/plot_{}_{}.pdf'.format(title,t_name), format='pdf')
    plt.show()







def plot_player_indice_utility_shape(indice : int, list_players : list[player_class],
                                     array_action : np.ndarray, array_sum_action : np.ndarray,
                                      temp :np.ndarray, figsize : tuple, name : str, print_title :bool = False, sav : bool = False):
    
    n,tmax = array_action.shape

    plt.figure(figsize=figsize, dpi=100, tight_layout=True)

    player = list_players[indice]
    player.reset_smc()
    # player.smc.reset_temperature_dynamic()

    for time in range(0,tmax-1):
        x = np.linspace(0,np.sum(player.action_set[1]))
        y = np.zeros_like(x)
        k=0
        
        for point in x:
            y[k] = player.utility_one_shot(point, array_sum_action[time] - array_action[indice][time])

            y[k] = y[k] + player.delta * (player.damage_function(array_sum_action[time], player.smc, player.GDP_max))**player.alpha
            k+=1
        player.update_smc(array_sum_action[time], E_EX[time], F_EX[time])
        if time%4==0:
            plt.plot(x, y , label=r'in {}'.format(2020 + time*5), alpha = 0.7)
            xmax = x[np.argmax(y)]
            ymax = y.max()
            plt.scatter( xmax, ymax, s=15)

    plt.ylabel(r'$u_{}(a_{})$'.format(indice+1, indice+1 ))
    plt.xlabel(r'$ a_{} \in \mathcal{{A}}_{}$'.format(indice+1,indice+1))
    if print_title :
        title = r'Shape of $u_{}(a_{}(t)) =$'.format(indice+1,indice+1)
        if player.benefit_function.coef[-1] !=0 :
            title = title+ r'${:.2f} a_{}(t)^2$'.format(player.benefit_function.coef[-1], indice+1)
        if player.benefit_function.coef[-2] < 0 :
            title = title+r'$ {:.2f} a_{}(t)$'.format(player.benefit_function.coef[-2], indice+1)
        if player.benefit_function.coef[-2] > 0 :
            if player.benefit_function.coef[-1] !=0 :
                title = title + r'$+$'
            title = title+ r'$ {:.2f} a_{}(t)$'.format(player.benefit_function.coef[-2], indice+1)
    
        # if player.damage_function.coef[-1] == 0 :
        #     title = title + r'$ - {} y_{{AT}}$'.format(player.damage_function.coef[-2])
        # if player.damage_function.coef[-2] == 0 :
        #     title = title + r'$ - {}y_{{AT}}^2 $'.format(player.damage_function.coef[-1])
        if player.damage_function.coef[-2] < 0 :
            title = title + r'$ - ({}y_{{AT}}^2 {} y_{{AT}})(a(t)) $'.format(player.damage_function.coef[-1],player.damage_function.coef[-2])
        if player.damage_function.coef[-2] > 0 :
            title = title + r'$ - ({}y_{{AT}}^2 + {} y_{{AT}})(a(t)) $'.format(player.damage_function.coef[-1],player.damage_function.coef[-2])
        if (player.alpha != 0) and (player.damage_function.coef[-1] != 0)  and (player.damage_function.coef[-2] != 0): 
            title = title + r'$^{{{}}}$'.format(player.alpha)
    plt.yticks([])
    # plt.xticks([player.action_set[0], player.action_set[1] /2, player.action_set[1]],labels=[r'0',r'$e^{{max}}_{}/2$'.format(indice+1), r'$e^{{max}}_{}$'.format(indice+1)])
    # plt.title(title)
    
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.legend(fancybox=True, framealpha=0.5)
    if sav:
        plt.savefig('plots/simu_individual_utilities_{}.pdf'.format(name), format='pdf')
    plt.show()
    if print_title :
        print(title)

def plot_player_indice_utility_shape_norescale(indice : int, list_players : list[player_class],
                                     array_action : np.ndarray, array_sum_action : np.ndarray,
                                      temp :np.ndarray, figsize : tuple, name : str, print_title :bool = False, sav : bool = False):
    
    n,tmax = array_action.shape

    plt.figure(figsize=figsize, dpi=100, tight_layout=True)

    player = list_players[indice]
    player.reset_smc()
    # player.smc.reset_temperature_dynamic()

    for time in range(0,tmax-1):
        x = np.linspace(0,np.sum(player.action_set[1]))
        y = np.zeros_like(x)
        k=0
        
        for point in x:
            y[k] = player.utility_one_shot(point, array_sum_action[time] - point)
            k+=1
        player.update_smc(array_sum_action[time], E_EX[time], F_EX[time])
        if time%4==0:
            plt.plot(x, y , label=r'in {}'.format(2020 + time*5), alpha = 0.7)
            xmax = x[np.argmax(y)]
            ymax = y.max()
            plt.scatter( xmax, ymax, s=15)

    plt.ylabel(r'$u_{}(a_{})$'.format(indice+1, indice+1 ))
    plt.xlabel(r'$ a_{} \in \mathcal{{A}}_{}$'.format(indice+1,indice+1))
    if print_title :
        title = r'Shape of $u_{}(a_{}(t)) =$'.format(indice+1,indice+1)
        if player.benefit_function.coef[-1] !=0 :
            title = title+ r'${:.2f} a_{}(t)^2$'.format(player.benefit_function.coef[-1], indice+1)
        if player.benefit_function.coef[-2] < 0 :
            title = title+r'$ {:.2f} a_{}(t)$'.format(player.benefit_function.coef[-2], indice+1)
        if player.benefit_function.coef[-2] > 0 :
            if player.benefit_function.coef[-1] !=0 :
                title = title + r'$+$'
            title = title+ r'$ {:.2f} a_{}(t)$'.format(player.benefit_function.coef[-2], indice+1)
    
        # if player.damage_function.coef[-1] == 0 :
        #     title = title + r'$ - {} y_{{AT}}$'.format(player.damage_function.coef[-2])
        # if player.damage_function.coef[-2] == 0 :
        #     title = title + r'$ - {}y_{{AT}}^2 $'.format(player.damage_function.coef[-1])
        if player.damage_function.coef[-2] < 0 :
            title = title + r'$ - ({}y_{{AT}}^2 {} y_{{AT}})(a(t)) $'.format(player.damage_function.coef[-1],player.damage_function.coef[-2])
        if player.damage_function.coef[-2] > 0 :
            title = title + r'$ - ({}y_{{AT}}^2 + {} y_{{AT}})(a(t)) $'.format(player.damage_function.coef[-1],player.damage_function.coef[-2])
        if (player.alpha != 0) and (player.damage_function.coef[-1] != 0)  and (player.damage_function.coef[-2] != 0): 
            title = title + r'$^{{{}}}$'.format(player.alpha)
    plt.yticks([])
    # plt.xticks([player.action_set[0], player.action_set[1] /2, player.action_set[1]],labels=[r'0',r'$e^{{max}}_{}/2$'.format(indice+1), r'$e^{{max}}_{}$'.format(indice+1)])
    # plt.title(title)
    
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.legend(fancybox=True, framealpha=0.5)
    if sav:
        plt.savefig('plots/simu_individual_utilities_{}.pdf'.format(name), format='pdf')
    plt.show()
    if print_title :
        print(title)



def plot_action_profile(list_players : list[player_class], array_action : np.ndarray, array_action_set :np.ndarray, figsize : tuple, name : str, sav : bool = False):

    n,tmax = array_action.shape

    plt.figure(figsize=figsize, dpi=100, tight_layout=True)

    for indice in range(n):
        p = plt.plot(2020 + np.arange(tmax)*5, array_action[indice], label='$a_{{{}}}(t)$, '.format( indice+1) + '$e^{{max}}_{} = {:d}$'.format(indice+1, int(list_players[indice].action_set[1])), alpha = 0.9)
        plt.plot(2020 + np.arange(tmax)*5, array_action_set[indice,:,1], ls='dotted', color=p[0].get_color(), alpha = 0.5,zorder=0)

    plt.ylabel('CO2 emission in GtC')
    plt.xlabel('Years')
    # plt.title('Players\' actions with respect to the state')
    plt.legend(ncol=2, loc='upper left',fancybox=True, framealpha=0.5)
    plt.xticks(2020 + np.arange(int(tmax//2)+1)*10, 2020 + np.arange(int(tmax//2)+1)*10, rotation=90)
    if sav :
        plt.savefig('plots/simu_actions_{}.pdf'.format(name), format='pdf')
    plt.show()

def plot_list_action_profile( list_array_action_with_label : np.ndarray, figsize : tuple, name : str, sav : bool = False):

    n,tmax = list_array_action_with_label[0][1].shape

    plt.figure(figsize=figsize, dpi=100, tight_layout=True)

    for idx in range(len(list_array_action_with_label)) :
        array_action_with_label = list_array_action_with_label[idx]
        label, array_action, array_action_set, ls, alpha, ls_e_max = array_action_with_label
        plt.gca().set_prop_cycle(None)
        for indice in range(n):
            p = plt.plot(2020 + np.arange(tmax)*5, array_action[indice], label='$a_{{{}}}(t)$ '.format(indice+1) + label, ls = ls, alpha = alpha)
            # plt.plot(2020 + np.arange(tmax)*5, array_action_set[indice,:,1], ls=ls_e_max, color=p[0].get_color(), alpha = 0.5,zorder=0)

    plt.ylabel('CO2 emission in GtC')
    plt.xlabel('Years')
    # plt.title('Players\' actions with respect to the state')
    plt.legend(ncol=2, loc='upper left',fancybox=True, framealpha=0.5)
    plt.xticks(2020 + np.arange(int(tmax//2)+1)*10, 2020 + np.arange(int(tmax//2)+1)*10, rotation=90)
    if sav :
        plt.savefig('plots/simu_actions_list_{}.pdf'.format(name), format='pdf')
    plt.show()

def plot_utilities(list_players : list[player_class], array_utilities : np.ndarray, figsize : tuple, name : str, sav : bool = False):

    n,tmax = array_utilities.shape

    plt.figure(figsize=figsize, dpi=100, tight_layout=True)

    for indice in range(n):
        plt.plot(2020 + np.arange(tmax)*5, array_utilities[indice], label='$u_{}(a_{}(t))$'.format(indice+1,indice+1), alpha = 0.9)

    plt.ylabel(r'Utilitiy in Trillions \$ 2011')
    plt.xlabel('Years')
    plt.xticks(2020 + np.arange(int(tmax//2)+1)*10, 2020 + np.arange(int(tmax//2)+1)*10, rotation=90)

    # plt.title('Utilities of players')
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.legend(ncol=2, loc='upper left',fancybox=True, framealpha=0.5)
    if sav :
        plt.savefig('plots/simu_utilities_{}.pdf'.format(name), format='pdf')
    plt.show()

def plot_list_utilities_profile( list_utilities_action_with_label : np.ndarray, figsize : tuple, name : str, sav : bool = False):

    n,tmax = list_utilities_action_with_label[0][1].shape

    plt.figure(figsize=figsize, dpi=100, tight_layout=True)

    for idx in range(len(list_utilities_action_with_label)) :
        array_utilities = list_utilities_action_with_label[idx]
        label, array_utilitie, ls, alpha= array_utilities
        plt.gca().set_prop_cycle(None)
        for indice in range(n):
            p = plt.plot(2020 + np.arange(tmax)*5, array_utilitie[indice], label='$u_{}$ '.format(indice+1) + label, ls = ls, alpha = alpha)

    plt.ylabel(r'Utilitiy in Trillions \$ 2011')
    plt.xlabel('Years')
    plt.xticks(2020 + np.arange(int(tmax//2)+1)*10, 2020 + np.arange(int(tmax//2)+1)*10, rotation=90)
    # plt.title('Players\' actions with respect to the state')
    plt.legend(ncol=2, loc='lower left',fancybox=True, framealpha=0.5)
    if sav :
        plt.savefig('plots/simu_utilities_list_{}.pdf'.format(name), format='pdf')
    plt.show()

def plot_temp_profile(temp :np.ndarray, figsize : tuple, name : str, temp_no_damage, sav : bool = False):

    tmax = len(temp)


    plt.figure(figsize=figsize, dpi=100, tight_layout=True)

    plt.plot(2020 + np.arange(tmax)*5, temp, label='Temperature')
    if type(temp_no_damage) == np.ndarray:
        plt.plot(2020 + np.arange(tmax)*5, temp_no_damage,'--',  label='Temperature without damage')

    plt.ylabel('Temperature variation from 1750 in °C')
    plt.xlabel('Years')

    # plt.title('Temperature evolution from 2020 to 2200')
    plt.legend( loc='upper left',fancybox=True, framealpha=0.5)

    plt.xticks(2020 + np.arange(int(tmax//2)+1)*10, 2020 + np.arange(int(tmax//2)+1)*10, rotation=90)
    if sav:
        plt.savefig('plots/simu_temp_{}.pdf'.format(name), format='pdf')
    plt.show()

def plot_list_temp_profile(list_temp_with_label :list[tuple], figsize : tuple, name : str, sav : bool = False ):

    tmax = len(list_temp_with_label[0][1])


    plt.figure(figsize=figsize, dpi=100, tight_layout=True)

    for indice in range(len(list_temp_with_label)) :
        label, temp, ls = list_temp_with_label[indice]
        plt.plot(2020 + np.arange(tmax)*5, temp, label='Temp. {}'.format(label), ls=ls)


    plt.ylabel('Temperature variation from 1750 in °C')
    plt.xlabel('Years')
    plt.xticks(2020 + np.arange(int(tmax//2)+1)*10, 2020 + np.arange(int(tmax//2)+1)*10, rotation=90)

    # plt.title('Temperature evolution from 2020 to 2200')
    plt.legend( loc='upper left',fancybox=True, framealpha=0.5)

    if sav:
        plt.savefig('plots/simu_temp_{}.pdf'.format(name), format='pdf')
    plt.show()

def plot_Game(game : Game, plot_nash : bool, plot_SO : bool, plot_bau : bool,  
                name :str, figsize:tuple = (8,4), indice : int = 0, print_title = False, sav : bool = False):
            
    list_action_with_label = []
    list_utilities_with_label = []
    list_temp_with_label = []

    if plot_nash:
        game.repeated_one_shot_game_NE()

        plot_utilities(game.list_players, game.ne_u_p, figsize, name, sav)
        # print('utilities')
        # plot_player_indice_utility_shape(indice,game.list_players, game.ne_a_p, game.ne_sum_a_p, game.ne_temp_p, figsize, name, print_title, sav)
        # print('utility shape')

        plot_action_profile(game.list_players, game.ne_a_p, game.ne_a_space_p, figsize, name, sav)
        # print('action profile')
        list_action_with_label.append(('NE', game.ne_a_p, game.ne_a_space_p, 'solid', 0.9, 'dotted'))
        list_utilities_with_label.append(('NE', game.ne_u_p, 'solid', 0.9))

        list_temp_with_label.append(('NE', game.ne_temp_p, '-'))
        # print('plot_nash')
    if plot_SO :
        game.repeated_one_shot_game_SO()

        plot_utilities(game.list_players, game.so_u_p, figsize, name+'_so', sav)
        # print('utilities')

        plot_action_profile(game.list_players, game.so_a_p, game.so_a_space_p, figsize, name+'_so', sav)
        # print('action profile')
        list_action_with_label.append(('SO', game.so_a_p, game.so_a_space_p, 'dashed', 0.6, 'dashdot'))
        list_utilities_with_label.append(('SO', game.so_u_p, 'dashed', 0.6))


        list_temp_with_label.append(('SO', game.so_temp_p, '--'))
        # print('plot_so')

    if plot_bau :
        game.repeated_one_shot_game_BAU()
        list_temp_with_label.append(('BAU', game.bau_temp_p, '-'))
        # print('plot_BAU')


    plot_list_temp_profile(list_temp_with_label, figsize, name, sav)
    plot_list_action_profile(list_action_with_label, figsize, name, sav)
    plot_list_utilities_profile(list_utilities_with_label, figsize, name, sav)


