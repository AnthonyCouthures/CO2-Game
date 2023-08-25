import numpy as np 
from functools import partial

FIG_SIZE=(9,6)

FIRST_YEAR = 2020
"Year of the first occurence of the game."

FINAL_YEAR = 2100
"Year of the last occurence of the game."

STEP = 5
"Number of years between two occurence of the game."

T = int((FINAL_YEAR - FIRST_YEAR) / STEP) 
r"Number of games played, given by int((FINAL_YEAR - FIRST_YEAR) / STEP) +1"


INCREASE_CO2_RATIO = 0.04467676003648222 
"Ratio of observed "


# SCM model

from models.geophysic_models import *


CARBON_MODEL = Carbon_JOOS()
"Default Carbon Model"

TEMPERATURE_MODEL = Temp_Discret_Geoffroy()
"Default Temperature Dynamic Model"

SCM = Simple_Climate_Model(CARBON_MODEL, TEMPERATURE_MODEL)
"Default Simple Climate Model"


#######################

from models.game_theory_model import *

N = 6
"Default number of players"

# Espaces d'action des joueurs en GtCO2


ACTION_SETS =  np.array([[[0.0, 11.47],        # China  
                         [0.0, 5.01],         # USA
                         [0.0, 3.14],         # EU
                         [0.0, 2.71],         # India
                         [0.0, 1.76],         # Russia
                         [0.0, 13.51]]]*2*T)*2       # other 

ACTION_SETS = np.swapaxes(ACTION_SETS,0,1)
# ACTION_SETS = np.array([[[ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ]],

#        [[11.29603892,  5.22830013,  3.75969311,  2.87500945,
#           1.55333567, 18.99997129],
#         [12.60117392,  6.00378399,  4.09964894,  3.66902644,
#           1.63743546, 21.49339715],
#         [13.90630892,  6.77926785,  4.43960476,  4.46304343,
#           1.72153524, 23.98682302],
#         [15.17796026,  7.60744754,  4.8503302 ,  5.51087533,
#           1.89714733, 26.99238004],
#         [16.44961161,  8.43562723,  5.26105564,  6.55870724,
#           2.07275942, 29.99793707],
#         [17.06651425,  9.35501414,  5.66857027,  7.83875885,
#           2.28563517, 33.8224664 ],
#         [17.68341689, 10.27440106,  6.07608489,  9.11881047,
#           2.49851091, 37.64699573],
#         [18.23582773, 11.41284042,  6.62503893, 10.48811467,
#           2.72379882, 42.33279249],
#         [18.78823857, 12.55127978,  7.17399297, 11.85741888,
#           2.94908673, 47.01858925],
#         [18.98638479, 13.23353773,  7.90045255, 13.18077282,
#           3.065076  , 52.20570386],
#         [19.18453101, 13.91579567,  8.62691213, 14.50412676,
#           3.18106526, 57.39281847],
#         [18.62749696, 14.58980466,  9.19749728, 15.68143965,
#           3.10399654, 62.02590708],
#         [18.0704629 , 15.26381366,  9.76808244, 16.85875253,
#           3.02692781, 66.65899568],
#         [16.66806015, 15.07856641, 10.02647529, 17.02222643,
#           2.80160576, 68.51470288],
#         [15.2656574 , 14.89331916, 10.28486814, 17.18570033,
#           2.57628371, 70.37041008],
#         [13.72093025, 14.43988751, 10.64855174, 16.53175468,
#           2.33267406, 70.75797603],
#         [12.1762031 , 13.98645586, 11.01223534, 15.87780903,
#           2.08906441, 71.14554198],
#         [12.1762031 , 13.98645586, 11.01223534, 15.87780903,
#           2.08906441, 71.14554198]]]).T

# repeated_arr = np.tile(ACTION_SETS[:, -1:, :], (1,100-T, 1))

# # concatenate the original array with the repeated array
# ACTION_SETS = np.concatenate([ACTION_SETS, repeated_arr], axis=1)


r"""Default action set of the players.

References
----------

.. [1] Hannah Ritchie, Max Roser and Pablo Rosado (2020) - "CO₂ and Greenhouse Gas Emissions". Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/co2-and-other-greenhouse-gas-emissions' [Online Resource]
       https://ourworldindata.org/grapher/annual-co2-emissions-per-country?facet=none&country=CHN~USA~European+Union+%2828%29~IND~RUS~Asia+%28excl.+China+and+India%29

"""

# ACTION_SETS =  np.array([[[0.0, 10.0],        # China  
#                          [0.0, 0.1],         # USA
#                          [0.0, 0.1],         # EU
#                          [0.0, 0.1],         # India
#                          [0.0, 0.1],         # Russia
#                          [0.0, 0.1]]]*T)  
# Coefficient d'impact de la temperature sur le climat 

DELTAS = np.array([1.1847,
                    1.1941,
                    1.1248,
                    0.9074,
                    1.2866,
                    1.1847])
r"""Default damage multiplier. 

References
----------

.. [1] http://www.fund-model.org/MimiFUND.jl/latest/tables/#Table-RT:-Regional-temperature-conversion-factor-1
"""



# PIB max des joueurs, j'ai pris le PIB 2020                

GDP_MAX = np.array([14.63,
                    19.29,
                    13.89,
                    2.5,
                    1.42,
                    26.27 - 14.63])*10
r"""Default maximum GDP 
References
----------

.. [1] https://ourworldindata.org/grapher/gross-domestic-product?tab=chart&time=latest&facet=none&country=CHN~USA~European+Union+%2828%29~IND~RUS~Asia+%28excl.+China+and+India%29~OWID_WRL~Africa~South+America+%28GCP%29~CAN~European+Union~East+Asia+and+Pacific
"""

NAMES = ['player {}'.format(i) for i in range(N)]
NAMES = ['China', 'USA', 'EU', 'India', 'Russia', 'ROTW']
"Names of the players"

BENEFITS = [benefit_affine for i in range(N)]
"Default list of benefit function, the benefit functions are affine."
BENEFITS_CONCAVE = [benefit_quadratic_concave for i in range(N)]
"Shortcut for a list of benefit function, the benefit functions are concave."
BENEFITS_SIGMOID = [benefit_sigm for i in range(N)]
"Shortcut for a list of benefit function, the benefit functions are sigmoidal"
BENEFITS_LOG = [benefit_log for i in range(N)]
BENEFITS_ROOT = [benefit_root for i in range(N)]
BENEFITS_ECO = [benefit_econmical_shape for i in range(N)]


BENEFITS_CONVEX = [benefit_quadratic_convex_with_percentage_green for i in range(N)]
"Shortcut for a list of benefit function, the benefit functions are convex."


DAMAGE =  partial(damage_polynome, coefficients = np.array([0]))
r"Default damage function, the damage function is expressed in \% of GDP loss."
ALPHA = 1
"Dafault alpha set to 1."

INCREASE_COEF_CO2_RATIO = [0.04467676003648222 for i in range(N)]
"Default increasing ratio of CO2 emission from a year to a other. currently wrong"

# INCREASE_COEF_CO2_RATIO[2] = 0

PERCENTAGES_GREEN = [0.0 for i in range(N)]
"Default list of the percentage of GDP of the players being noncarbonated."

PERCENTAGE_GDP = True
"Boolean. If True, the damage are expressed as loss in percentage of GDP."

DISCOUNT = 1

TARGET_TEMPERATURE = 2

FINAL_MULTIPLIER = -20