import numpy as np 


FIG_SIZE=(9,6)

FIRST_YEAR = 2020
"Year of the first occurence of the game."

FINAL_YEAR = 2100
"Year of the last occurence of the game."

STEP = 5
"Number of years between two occurence of the game."

T = int((FINAL_YEAR - FIRST_YEAR) / STEP) +1
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


ACTION_SETS = np.array([[0.0, 11.47],        # China  
                        [0.0, 5.01],         # USA
                        [0.0, 3.14],         # EU
                        [0.0, 2.71],         # India
                        [0.0, 1.76],         # Russia
                        [0.0, 7.51]])        # other Asia  
r"""Default action set of the players.

References
----------

.. [1] Hannah Ritchie, Max Roser and Pablo Rosado (2020) - "CO₂ and Greenhouse Gas Emissions". Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/co2-and-other-greenhouse-gas-emissions' [Online Resource]
       https://ourworldindata.org/grapher/annual-co2-emissions-per-country?facet=none&country=CHN~USA~European+Union+%2828%29~IND~RUS~Asia+%28excl.+China+and+India%29

"""
# Coefficient d'impact de la temperature sur le climat 

DELTAS = np.array([1.1847,
                    1.1941,
                    1.1248,
                    0.9074,
                    1.2866,
                    1.1847])**2
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
                    26.27 - 14.63])
r"""Default maximum GDP 
References
----------

.. [1] https://ourworldindata.org/grapher/gross-domestic-product?tab=chart&time=latest&facet=none&country=CHN~USA~European+Union+%2828%29~IND~RUS~Asia+%28excl.+China+and+India%29~OWID_WRL~Africa~South+America+%28GCP%29~CAN~European+Union~East+Asia+and+Pacific
"""

NAMES = ['player {}'.format(i) for i in range(N)]
"Names of the players"

BENEFITS = [benefit_affine for i in range(N)]
"Default list of benefit function, the benefit functions are affine."
BENEFITS_CONCAVE = [benefit_quadratic_concave for i in range(N)]
"Shortcut for a list of benefit function, the benefit functions are concave."
BENEFITS_SIGMOID = [benefit_sigm for i in range(N)]
"Shortcut for a list of benefit function, the benefit functions are sigmoidal"


BENEFITS_CONVEX = [benefit_quadratic_convex_with_percentage_green for i in range(N)]
"Shortcut for a list of benefit function, the benefit functions are convex."


DAMAGE =  damage_polynome(np.array([0]))
r"Default damage function, the damage function is expressed in \% of GDP loss."
ALPHA = 1
"Dafault alpha set to 1."

INCREASE_COEF_CO2_RATIO = [0.04467676003648222 for i in range(N)]
"Default increasing ratio of CO2 emission from a year to a other. currently wrong"

INCREASE_COEF_CO2_RATIO[2] = 0

PERCENTAGES_GREEN = [0 for i in range(N)]
"Default list of the percentage of GDP of the players being noncarbonated."

PERCENTAGE_GDP = True
"Boolean. If True, the damage are expressed as loss in percentage of GDP."