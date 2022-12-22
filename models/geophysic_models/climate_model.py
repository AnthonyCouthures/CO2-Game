r"""The Simple Climate Model (SCM) object is composed of a Carbon Model, a Radiative Forcing Formula and a Temperature Dynamic. 

The idea of the SCM is to have a object to which we can provide some CO2 emission and being able to predict a atmospheric temperature after some time.

"""


from models.geophysic_models import *
import numpy as np
from copy import deepcopy


class Simple_Climate_Model :
    """Class joining the Carbon Cycle model, the radiative forcing and the temperature dynamic together. 

    Parameters
    ----------
    carbon_model : Linear_Carbon_Model
        The carbon model use in the geophysic model.
    temperature_model : Linear_Temperature_Dynamic
        The temperature dynamic model use in the geophysic model.
    non_human_carbon_emission : np.ndarray, optional
        Exogeneous carbon emissions, by default E_EX
    non_co2_radiative_forcing : np.ndarray, optional
        Exogeneous radiative forcing, by default F_EX
    """

    def __init__(self, carbon_model : Linear_Carbon_Model, temperature_model : Linear_Temperature_Dynamic,
                non_human_carbon_emission = E_EX, non_co2_radiative_forcing = F_EX) -> None:


        # Number of cycle 

        self.num_cycle = 0
        "Number of the current cycle. i.e for taking the correspondant exogeneous emissiosn and radiative forcing."

        # Carbon Cycle initialization
        self.carbon_model = carbon_model
        self.carbon_state = deepcopy(self.carbon_model.initial_state)
        "Curent carbon state of the model"
        self.non_human_carbon_emission = non_human_carbon_emission
        self.atmospheric_carbon = self.carbon_model.atmospheric_carbon(self.carbon_state)
        "Current atmospheric carbon of the model"

        # Radiative Forcing
        self.radiative_forcing_function = radiative_forcing_function
        self.non_co2_radiative_forcing = non_co2_radiative_forcing
        self.radiative_forcing = radiative_forcing_function(self.atmospheric_carbon) # + self.non_co2_radiative_forcing[0]
        
        
        # Temperature dynamic initialization 
        self.temperature_model = temperature_model
        self.temperature_state = deepcopy(self.temperature_model.initial_state)
        "Curent temperature state of the model"

        self.atmospheric_temp = self.temperature_model.atmospheric_temperature(self.temperature_state)
        "Current atmospheric temperature of the model"


    def __copy__(self):
        return type(self)(self.carbon_model, 
                            self.temperature_model)

    def __deepcopy__(self, memo): 
        id_self = id(self)        
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                        deepcopy(self.carbon_model, memo),
                        deepcopy(self.temperature_model, memo),
                        )
            memo[id_self] = _copy 
        return _copy

 
    def reset_carbon_model(self) -> None:
        """Function which reset the carbon model.
        """
        self.carbon_state = deepcopy(self.carbon_model.initial_state)

    def reset_temperature_dynamic(self) -> None:
        """Function which reset the temperature model.
        """
        self.temperature_state = deepcopy(self.temperature_model.initial_state)

    def reset(self) -> None :
        """Function which reset the Simple Climate Model.
        """
        self.num_cycle = 0
        self.reset_carbon_model()
        self.reset_temperature_dynamic()


    def initialize_carbon_model(self, initial_carbon) -> None:
        """Initialization of the carbon model.

        Parameters
        ----------
        initial_carbon : np.ndarray
            A initial state for the carbon model.
        """
        self.carbon_state = deepcopy(initial_carbon)
        self.atmospheric_carbon = self.carbon_model.atmospheric_carbon(self.carbon_state)
        
    def initialize_temperature_dynamic(self, initial_temperature):
        """Initialization of the temperature model.

        Parameters
        ----------
        initial_temperature : np.ndarray
            A initial state for the temperature model.
        """
        self.temperature_state = deepcopy(initial_temperature)
        self.atmospheric_temp = self.temperature_model.atmospheric_temperature(self.temperature_state)
        
    def initialize(self, initial_carbon, initial_temperature, num_cycle):
        """Function that initialize the Simple Climate Model.

        Parameters
        ----------
        initial_carbon : np.ndarray
            A initial state for the carbon model.
        initial_temperature : np.ndarray
            A initial state for the temperature model.
        num_cycle : int
            A initial number of cycle.
        """
        self.initialize_carbon_model(initial_carbon)
        self.initialize_temperature_dynamic(initial_temperature)
        self.num_cycle = num_cycle

    def five_years_cycle_deep(self, humans_emission : float, exogeneous_emission : float = None,
                            exogeneous_radiative_forcing : float = None) -> None : 
        """Function that make a five years cycle of the Simple Climate Model for given parameters. 

        Parameters
        ----------
        humans_emission : float
            The humans emissions in GtCO2/y
        exogeneous_emission : float, optional
            The exogeneous emissions of CO2 in GtCO2/y, by default None
        exogeneous_radiative_forcing : float, optional
            The exogenerous radiative forcing in W/m²/y, by default None
        """

        if exogeneous_emission == None :
            exogeneous_emission = self.non_human_carbon_emission[self.num_cycle]
        if exogeneous_radiative_forcing == None :
            exogeneous_radiative_forcing = self.non_co2_radiative_forcing[self.num_cycle]

        # Carbon Cycle
        emission = humans_emission + exogeneous_emission
        self.carbon_state = self.carbon_model.five_years_cycle(emission, self.carbon_state)
        self.atmospheric_carbon = self.carbon_model.atmospheric_carbon(self.carbon_state)

        # Radiative Forcing
        self.co2_radiative_forcing = self.radiative_forcing_function(self.atmospheric_carbon)
        self.radiative_forcing = self.co2_radiative_forcing + exogeneous_radiative_forcing

        # Temperature dynamic
        self.temperature_state = self.temperature_model.five_years_cycle(self.radiative_forcing, self.temperature_state)
        self.atmospheric_temp = self.temperature_model.atmospheric_temperature(self.temperature_state)

        self.num_cycle = self.num_cycle + 1


    def five_years_atmospheric_temp(self, humans_emission : float, exogeneous_emission : float = None,
                            exogeneous_radiative_forcing : float = None) -> float : 
        """Function that give a projection of the mean average atmospheric temperature after five years. 

        Parameters
        ----------
        humans_emission : float
            The humans emissions in GtCO2/y
        exogeneous_emission : float, optional
            The exogeneous emissions of CO2 in GtCO2/y, by default None
        exogeneous_radiative_forcing : float, optional
            The exogenerous radiative forcing in W/m²/y, by default None
        Returns
        -------
        float
            Mean average atmospheric temperature after five years in °C.        
    
        """

        if exogeneous_emission == None :
            exogeneous_emission = self.non_human_carbon_emission[self.num_cycle]
        if exogeneous_radiative_forcing == None :
            exogeneous_radiative_forcing = self.non_co2_radiative_forcing[self.num_cycle]

        # Carbon Cycle
        emission = humans_emission + exogeneous_emission
        carbon_state = self.carbon_model.five_years_cycle(emission, self.carbon_state)
        atmospheric_carbon = self.carbon_model.atmospheric_carbon(carbon_state)

        # Radiative Forcing
        co2_radiative_forcing = self.radiative_forcing_function(atmospheric_carbon)
        radiative_forcing = co2_radiative_forcing + exogeneous_radiative_forcing

        # Temperature dynamic
        temperature_state = self.temperature_model.five_years_cycle(radiative_forcing, self.temperature_state)
        atmospheric_temp = self.temperature_model.atmospheric_temperature(temperature_state)

        return atmospheric_temp


    def five_years_atmospheric_carbon(self, humans_emission : np.ndarray, exogeneous_emission : np.ndarray = None) -> float :
        """Function that give a projection of the quantity of carbon in the atmosphere in GtC after five years. 

        Parameters
        ----------
        humans_emission : float
            The humans emissions in GtCO2/y
        exogeneous_emission : float, optional
            The exogeneous emissions of CO2 in GtCO2/y, by default None

        Returns
        -------
        float
            Quantity of carbon in the atmosphere in GtC.
        """

        if exogeneous_emission == None :
            exogeneous_emission = self.non_human_carbon_emission[self.num_cycle]
        if exogeneous_radiative_forcing == None :
            exogeneous_radiative_forcing = self.non_co2_radiative_forcing[self.num_cycle]

        # Carbon Cycle
        emission = humans_emission + exogeneous_emission
        carbon_state = self.carbon_model.five_years_cycle(emission, self.carbon_state)
        atmospheric_carbon = self.carbon_model.atmospheric_carbon(carbon_state)

        return atmospheric_carbon



    def multiple_cycles(self, num_cycles : int,
                            array_humans_emission : np.ndarray, array_exogeneous_emission : np.ndarray = None,
                            array_exogeneous_radiative_forcing : np.ndarray = None) -> tuple :

        if array_exogeneous_emission == None :
            array_exogeneous_emission = self.non_human_carbon_emission
        if array_exogeneous_radiative_forcing == None :
            array_exogeneous_radiative_forcing = self.non_co2_radiative_forcing

        list_carbon = [np.copy(self.carbon_state)]
        list_forcing = [np.copy(self.radiative_forcing)]                
        list_temperature = [np.copy(self.temperature_state)]
        list_atmospheric_temp = [np.copy(self.atmospheric_temp)]

        for indice in range(num_cycles-1):
            self.five_years_cycle_deep(array_humans_emission[indice], array_exogeneous_emission[indice],
                        array_exogeneous_radiative_forcing[indice])

            # print(self.carbon)
            list_carbon.append(np.copy(self.carbon_state))
            # print(list_carbon)
            list_forcing.append(np.copy(self.radiative_forcing))
            # print(list_forcing)
            list_temperature.append(np.copy(self.temperature_state))
            # print(list_temperature)
            list_atmospheric_temp.append(np.copy(self.atmospheric_temp))

        array_carbon = np.vstack(list_carbon).T
        array_forcing = np.array(list_forcing)
        array_temperature = np.vstack(list_temperature).T
        array_atmospheric_temp = np.vstack(list_atmospheric_temp)
        # print('array_carbon : ', array_carbon)
        # print('array_forcing : ', array_forcing)
        # print('array_temperature : ' , array_temperature)
        # print('array_atmospheric_temp : ', array_atmospheric_temp)
        return array_carbon, array_forcing, array_temperature, array_atmospheric_temp

        
