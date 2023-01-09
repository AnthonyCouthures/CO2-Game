import numpy as np
from parameters import *
from scipy.misc import *

def evalutate_linear_model(x, A, b, d, tmax, initial_state):
    vecteur = []
    state = initial_state
    for t in range(tmax):
        state = A @ state + d * x[t]
        vecteur.append(b @ state)
    return np.array(vecteur)

def derivative_damage(x, damage_func):
    return derivative(damage_func, x, order=5, dx=1e-6)

def jacobian_damage_function(x, damage_func):
    return np.array(derivative_damage(x, damage_func))

def jacobian_sum(t_periode, n_player):
        
    matrice = np.zeros((t_periode, n_player * t_periode))
    for i in range(n_player * t_periode):
        k = i//n_player
        matrice[k,i] = 1

    return matrice

def jacobian_linear_model(A, b, d, tmax):
    T_matrix = np.diag(np.repeat(np.dot(b, d), tmax))
    for i in range(tmax):
        for j in range(i):
            T_matrix[i,j] = np.dot(b, np.linalg.matrix_power(A, i-j) @ d) # Eléments au-dessous de la diagonale
    return T_matrix

def jacobian_forcing(atmospheric_carbon : np.ndarray, Fx2CO2 = F_2XCO2):
    return np.diag(atmospheric_carbon**(-1)) * Fx2CO2 / np.log(2)




def jacobian_damage_all_players(a : np.ndarray, damage_func : damage_function,  SCM : Simple_Climate_Model):

    n_player, t_periode = a.shape
    sum_a = np.sum(a, axis=0)

    CC = SCM.carbon_model
    TD = SCM.temperature_model

    carbon_AT = evalutate_linear_model(sum_a, CC.Ac, CC.bc, CC.dc, t_periode, SCM.carbon_state)
    if CC.name == 'Joos-et-al':
        carbon_AT += C_1750
    forcing = radiative_forcing_function(carbon_AT) + F_EX[:t_periode]
    temperature_AT = evalutate_linear_model(forcing, TD.At, TD.bt, TD.dt, t_periode, SCM.temperature_state)

    jac_carbon_AT = jacobian_linear_model(CC.Ac, CC.bc, CC.dc, t_periode)
    jac_forcing = jacobian_forcing(carbon_AT)
    jac_temperature_AT = jacobian_linear_model(TD.At, TD.bt, TD.dt, t_periode)
    # jac_damage = np.tile(jacobian_damage_function(temperature_AT, damage_func),[n_player,1])
    jac_damage = jacobian_damage_function(temperature_AT, damage_func)
    jac_sum = jacobian_sum(t_periode, n_player)
    
    jacobian =  jac_damage @ jac_temperature_AT @ jac_forcing @ jac_carbon_AT @ jac_sum
    # jacobian = np.kron(jacobian_one_player, np.ones(N))
    # jacobian = jacobian_one_player.repeat(n_player)

    return jacobian


# evalutate_linear_model = memory.cache(evalutate_linear_model)


def potential(n_player, t_periode, a : np.ndarray, list_benefit : list[callable], list_weight : list[float], damage_func : damage_function, SCM : Simple_Climate_Model):

    a = np.reshape(a, (n_player,t_periode), 'F' )
    sum_a = np.sum(a, axis=0)

    CC = SCM.carbon_model
    TD = SCM.temperature_model

    carbon_AT = evalutate_linear_model(sum_a, CC.Ac, CC.bc, CC.dc, t_periode, SCM.carbon_state) 
    if CC.name == 'Joos-et-al':
        carbon_AT += C_1750
    forcing = radiative_forcing_function(carbon_AT) + F_EX[:t_periode]
    temperature_AT = evalutate_linear_model(forcing, TD.At, TD.bt, TD.dt, t_periode, SCM.temperature_state)

    benefit = sum([1/DELTAS[j] *  sum([list_benefit[j](GDP_MAX[j], ACTION_SETS[j,1])(a[j,i]) for i in range(t_periode)]) for j in range(n_player)])
    damage = sum(damage_func(temperature_AT))
    value = benefit - damage

        
    return value

def jacobian_benefice(a : np.ndarray, list_benefit : list[callable], list_weight : list[float]):
    n_player, t_periode = a.shape

    diag = [1/list_weight[j] *  derivative(list_benefit[j](GDP_MAX[j], ACTION_SETS[j,1]),a[j,i]) for i in range(t_periode) for j in range(n_player)]
    return np.array(diag).T

def jacobian_potential(n_player, t_periode, a : np.ndarray, list_benefit : list[callable], list_weight : list[float], damage_func : callable, scm : Simple_Climate_Model):
    a = np.reshape(a,(n_player, t_periode),'F') # the reshape function is weird it's important
    return jacobian_benefice(a, list_benefit, list_weight) - jacobian_damage_all_players(a, damage_func, scm)


def eval_scm(path, SCM : Simple_Climate_Model):
    CC = SCM.carbon_model
    TD = SCM.temperature_model

    carbon_AT = evalutate_linear_model(path, CC.Ac, CC.bc, CC.dc, T, CC.initial_state) 
    if CC.name == 'Joos-et-al':
        carbon_AT += C_1750
    forcing = radiative_forcing_function(carbon_AT) + F_EX[:len(path)]
    temperature_AT = evalutate_linear_model(forcing, TD.At, TD.bt, TD.dt, T, TD.initial_state)
    return temperature_AT, carbon_AT
    