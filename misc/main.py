import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.geophysic_models import *

from models.geophysic_models.carbon_cycle_models import *
from models.geophysic_models.temperature_dynamic_model import *
from models.geophysic_models.radiative_forcing import *

from parameters import *
from plot_function import *


carbon_model = Carbon_FUND
radiative_forcing_equation = radiative_forcing_function
temp_model = Temp_Discret_Geoffroy

# Joos & al. initial concentration
# initial_carbon_concentration = np.array([139.1, 90.2, 29.2, 4.2])
equilibrium_atmospheric_co2 = 588

initial_tempetature = [0.85,0.0068]


array_emission = np.ones(T)*30

RCP = pd.read_csv('data/rcp_data.csv',sep=",")             # Global Budget for all player in MtC
list_s = [np.array(RCP.loc[i][6:6+T]) * 44/12 for i in range(4)]
list_label = [RCP.loc[i][1] for i in range(4)]

for index in range(len(list_s)):
    title = list_label[index]
    array_emission = list_s[index]

    list_carbon_model = [Carbon_FUND(), Carbon_DICE_2013R(), Carbon_DICE_2016R(), Carbon_JOOS()]
    list_temp_model = [Temp_DICE_2016R()]
    plot_list_atmospheric_temperature(array_emission, list_carbon_model, list_temp_model, title=title, save=True)

    list_temp_model = [Temp_Discret_Geoffroy()]
    plot_list_atmospheric_temperature(array_emission, list_carbon_model, list_temp_model, title=title, save=True)

    # plot_list_surface_temperature(array_emission, list_carbon_model, radiative_forcing_equation, list_temp_model ,
    #                                         initial_tempetature, title=title, save=True)