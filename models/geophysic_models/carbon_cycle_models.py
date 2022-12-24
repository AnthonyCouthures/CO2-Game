r"""Carbon Modeling structure.

The base model is discrete and linear.  

Linear models of carbon cycle are mostly represented as flows of carbon between :math:`m \in \mathbb{N}` different boxes or reservoirs. They are modeled as a system of :math:`m \in \mathbb{N}` equations as the following 

.. math::

    C(t) &= A_C C(t-1) + d_C E(t) , \quad \forall t \in \mathbb{N} \\
    C(0) &= C_0

Where,

- :math:`C(t)` is the `m`-vector of carbon concentration in the reservoirs at time :math:`t`,

- :math:`E(t)` is the CO2 emissions at time :math:`t`,

- :math:`A_C` is the square matrix of constants that represent the carbon flux between the boxes,

- :math:`d_C` is the vector that affects the emissions to the boxes.

The number of boxes, the matrix :math:`A_C` and the vector :math:`d_C` differ from a model to another.  

The atmospheric Carbon concentration is given by,

.. math::

    C_{AT}(t)=b_C^{\mathrm{T}} C(t), \quad \forall t \in \mathbb{N}

Where :math:`b_C` is the vector that will represent the atmospheric component of the CO2 concentration vector :math:`C(t)`. In fact, :math:`b_C` will be fixed as well, in each of the following models.

"""



import numpy as np
from .constants import C_1750
from .constants import ppm_co2_to_c, ppm_co2_to_co2, co2_to_C
from scipy.linalg import expm
from .usefull_method import exact_discretization


class Linear_Carbon_Model :
    """Structure of a Linear Carbon Model.

    """

    name : str = " "
    "Name of the model"
    timestep : int = 1  
    "Time-step of the model"
    Ac : np.ndarray = np.ones(1) 
    "Transition matrix between the differents boxes"
    dc : np.ndarray = np.ones(1) 
    "Transition vector from the emission to the boxes"
    bc : np.ndarray = np.ones(1) 
    "Transition vector from the boxes to the atmospheric layer"
    bdc = bc @ dc
    initial_state : np.ndarray = np.ones(1) 
    "Initial content of the boxes"


    def __copy__(self):
        return type(self)(self.Ac, self.dc, self.bc, self.initial_state, self.name)

    def __deepcopy__(self, memo): 
        id_self = id(self)        
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)()
            memo[id_self] = _copy 
        return _copy
    
    
    def init_carbon_cycle(self, initial_state : np.ndarray) -> None:
        r"""Function which initialize the Carbon Cycle from a given state.

        Parameters
        ----------
        initial_state : np.ndarray
            Intial state of the carbon cycle.
        """
        
        self.initial_state = initial_state

    def five_years_cycle(self, emission : np.ndarray) -> None:
        r"""Function which return the Carbon state in GtC after five years for a given emission per years.

        Parameters
        ----------
            emission (np.ndarray): Emission over the five years in GtCO2/year

        Returns
        -------
            np.ndarray: Carbon in the boxes after 5 years GtC
        """
        pass

    def n_years_cycle(self, n : int, emission : np.ndarray) -> None:
        """Function which return the Carbon state in GtC after :math:`n` years for a given emission per years.

        Parameters
        ----------
        n : int
            Numbers of years
        emission : np.ndarray
            Emission over the five years in GtCO2/year
        """
        pass

    def atmospheric_carbon(self, state) -> float:
        r"""Function which return the Carbon in the atmosphere in GtC from a state.

        Returns
        -------
        atmospheric_carbon : float
            Carbon in the atmosphere in GtC.
        """

        return self.bc @ state 

    def five_years_atmospheric_carbon(self, emission : float, state : np.ndarray) -> float:
        r"""Function which return the Carbon in the atmosphere in GtC after five years for a given emission.

        Parameters
        ----------
        emission : float
            Emissions over the five years in GtCO2/year.
        state : np.ndarray
            A state of the carbon cycle.

        Returns
        -------
        atmospheric_carbon : float
            Carbon in the atmosphere in GtC.
        """

        return self.bc @ self.Ac @ state + self.bdc * emission

    def n_years_atmospheric_carbon(self,n :int, emission : float, state : np.ndarray) -> float:
        r"""Function which return the Carbon in the atmosphere in GtC after five years for a given emission.

        Parameters
        ----------
        n : int
            Numbers of years
        emission : float
            Emissions over the five years in GtCO2/year.
        state : np.ndarray
            A state of the carbon cycle.

        Returns
        -------
        atmospheric_carbon : float
            Carbon in the atmosphere in GtC.
        """

        state = self.n_years_cycle(n, emission, state)
        return self.atmospheric_carbon(state)

class Carbon_FUND(Linear_Carbon_Model) :
    r""" Carbon model used in the FUND IAM.
    
    Notes
    -----
    This Carbon model initialy take emission in ppm and return atmospheric CO2 in ppm. We modified it to take 
    emission in GtCO2 and return CtC. 

    For initialization we use the the jupyternotebook provided by FUND avalaible at https://github.com/fund-model/MimiFUND.jl/blob/master/calibration/co2cycle/fund_co2_cycle.ipynb
    for the year 2020. 

    References
    ----------


    """

    name : str = 'FUND'
    r"Name of the model"
    timestep : int = 1  
    r"Time-step of the model"
    Ac : np.ndarray = np.diag(np.exp(-np.concatenate((np.zeros(1), np.array([363,74,17,2], dtype=float)**(-1))))) 
    r"Transition matrix between the differents boxes, by default :math:`\begin{bmatrix}   1. & 0. & 0. & 0. & 0.\\   0. & 0.99724897 & 0. & 0. & 0.\\   0. & 0. & 0.98657738 & 0. & 0.\\   0. & 0. & 0. & 0.94287314 & 0.\\   0. & 0. & 0. & 0. & 0.60653066\\ \end{bmatrix}`"
    dc : np.ndarray = np.array([0.13,    0.20,   0.32,   0.25,   0.1]) * 1/ppm_co2_to_co2 
    r"Transition vector from the emission to the boxes, by default :math:`\begin{bmatrix}   0.13\\   0.2\\   0.32\\   0.25\\   0.1\\ \end{bmatrix} \times \frac{1}{ppm\_co2\_to\_co2}`"
    bc : np.ndarray = np.ones(5) * ppm_co2_to_c 
    r"Transition vector from the boxes to the atmospheric layer, by default :math:`\begin{bmatrix}1. & 1. & 1. & 1. & 1.\\ \end{bmatrix} \times ppm\_co2\_to\_c`"
    initial_state : np.ndarray = np.array([320.34, 36.96, 43.49, 15.80, 1.23]) 
    r"Initial content of the boxes, by default :math:`\begin{bmatrix}   320.34\\   36.96\\   43.49\\   15.80\\ 1.23\\ \end{bmatrix}`"


    def five_years_cycle(self, emission : np.ndarray, state : np.ndarray) -> np.ndarray:
        r"""Function that calculate one cycle of carbon for a timestep of 5 years with the FUND carbon cycle with zero order hold assumption. 

        Parameters
        ----------
            emission (np.ndarray): Emission over the five years in GtCO2/year

        Returns
        -------
            np.ndarray: Carbon in the boxes after 5 years GtC
        """
        for t in range(int(5 / self.timestep)):
            state = self.Ac @ state + self.dc * emission
        return state
        


class Carbon_DICE_2013R(Linear_Carbon_Model) :
    r"""Carbon Cycle from DICE-2013 model. 



    References
    ----------

    .. [1] Nordhaus, William. « Estimates of the Social Cost of Carbon: Concepts and Results from the DICE-2013R Model and Alternative Approaches». Journal of the Association of Environmental and Resource Economists 1, nᵒ 1/2 (mars 2014): 273-312. https://doi.org/10.1086/676035.
    .. [2] Kellett, Christopher M., Steven R. Weller, Timm Faulwasser, Lars Grüne, et Willi Semmler. « Feedback, Dynamics, and Optimal Control in Climate Economics». Annual Reviews in Control 47 (1 janvier 2019): 7-20. https://doi.org/10.1016/j.arcontrol.2019.04.003.
    .. [3] Rezai, Armon, Simon Dietz, Frederick van der Ploeg, et Frank Venmans. « Are Economists Getting Climate Dynamics Right and Does It Matter? » Copernicus Meetings, 9 mars 2020. https://doi.org/10.5194/egusphere-egu2020-20039.


    """

    name : str = 'DICE-2013R'
    r"Name of the model"
    timestep : int = 5  
    r"Time-step of the model"
    Ac : np.ndarray =  np.array([[0.912,  0.03833,  0],
                            [0.088,  0.9592,  0.0003375],
                            [0,     0.00250,  0.9996625]])
    r"Transition matrix between the differents boxes, by default :math:`\begin{bmatrix}   0.912 & 0.03833 & 0 \\   0.088 & 0.9592 & 0.0003375\\   0. & 0.0025 & 0.9996625\\ \end{bmatrix}`"
    dc : np.ndarray = np.array([1,   0,  0]) * co2_to_C
    r"Transition vector from the emission to the boxes, by default :math:`\begin{bmatrix}   1\\   0\\   0\\ \end{bmatrix} \times co2\_to\_C`"
    bc : np.ndarray = np.array([1,   0,  0])
    r"Transition vector from the boxes to the atmospheric layer, by default :math:`\begin{bmatrix}1 & 0 & 0 \\ \end{bmatrix}`"
    initial_state : np.ndarray = np.array([830.4, 1527, 10010]) 
    r"Initial content of the boxes, by default :math:`\begin{bmatrix}   830.4\\   1527\\   10010\\ \end{bmatrix}`"


       


    def five_years_cycle(self, emission : np.ndarray, state : np.ndarray) -> np.ndarray:
        r"""Function which return the Carbon in the atmosphere in GtC after five years for a given emission.

        Parameters
        ----------
        emission : float
            Emissions over the five years in GtCO2/year.
        state : np.ndarray
            A state of the carbon cycle.

        Returns
        -------
        atmospheric_carbon : float
            Carbon in the atmosphere in GtC.
        """
        state = self.Ac @ state + self.dc * emission
        return state


class Carbon_DICE_2016R(Linear_Carbon_Model) :
    r"""Carbon Cycle from DICE-2016 model. 

    References
    ----------

    .. [1] Nordhaus, William. « Estimates of the Social Cost of Carbon: Concepts and Results from the DICE-2013R Model and Alternative Approaches ». Journal of the Association of Environmental and Resource Economists 1, nᵒ 1/2 (mars 2014): 273-312. https://doi.org/10.1086/676035.
    .. [2] Rezai, Armon, Simon Dietz, Frederick van der Ploeg, et Frank Venmans. « Are Economists Getting Climate Dynamics Right and Does It Matter? » Copernicus Meetings, 9 mars 2020. https://doi.org/10.5194/egusphere-egu2020-20039.


    """

    name : str = 'DICE-2016R'
    r"Name of the model"
    timestep : int = 5  
    r"Time-step of the model"
    Ac : np.ndarray =  np.array([[0.88,  0.196,  0],
                            [0.12,  0.797,  0.001465],
                            [0,     0.007,  0.998535]])
    r"Transition matrix between the differents boxes, by default :math:`\begin{bmatrix}   0.88 & 0.196 & 0.\\   0.12 & 0.797 & 0.001465\\   0. & 0.007 & 0.998535\\ \end{bmatrix}`"
    dc : np.ndarray = np.array([1,   0,  0]) * co2_to_C
    r"Transition vector from the emission to the boxes, by default :math:`\begin{bmatrix}   1\\   0\\   0\\ \end{bmatrix} \times co2\_to\_C`"
    bc : np.ndarray = np.array([1,   0,  0])
    r"Transition vector from the boxes to the atmospheric layer, by default :math:`\begin{bmatrix}1 & 0 & 0 \\ \end{bmatrix}`"
    initial_state : np.ndarray = np.array([851, 460, 1740]) 
    r"Initial content of the boxes, by default :math:`\begin{bmatrix}   851\\   460\\   1740\\ \end{bmatrix}`"





    def five_years_cycle(self, emission : np.ndarray, state : np.ndarray) -> np.ndarray:
        r"""Function which return the Carbon in the atmosphere in GtC after five years for a given emission.

        Parameters
        ----------
        emission : float
            Emissions over the five years in GtCO2/year.
        state : np.ndarray
            A state of the carbon cycle.

        Returns
        -------
        atmospheric_carbon : float
            Carbon in the atmosphere in GtC.
        """
        state = self.Ac @ state + self.dc * emission
        return state
    




class Carbon_JOOS(Linear_Carbon_Model) :
    r"""Carbon Cycle from Joos et al.. 

    References
    ----------

    .. [1] Joos, F., R. Roth, J. S. Fuglestvedt, G. P. Peters, I. G. Enting, W. von Bloh, V. Brovkin, et al. « Carbon Dioxide and Climate Impulse Response Functions for the Computation of Greenhouse Gas Metrics: A Multi-Model Analysis ». Atmospheric Chemistry and Physics 13, nᵒ 5 (8 mars 2013): 2793-2825. https://doi.org/10.5194/acp-13-2793-2013.
    .. [2] Rezai, Armon, Simon Dietz, Frederick van der Ploeg, et Frank Venmans. « Are Economists Getting Climate Dynamics Right and Does It Matter? » Copernicus Meetings, 9 mars 2020. https://doi.org/10.5194/egusphere-egu2020-20039.

    """

    name : str = 'Joos-et-al'
    r"Name of the model"
    timestep : int = 1  
    r"Time-step of the model"
    Ac : np.ndarray = np.array([[1, 0,      0,      0],
                                [0, 0.9975, 0,      0],
                                [0, 0,      0.9730, 0],
                                [0, 0,      0,      0.7927]])
    r"Transition matrix between the differents boxes, by default :math:`\begin{bmatrix} 1. & 0. & 0. & 0.\\ 0. & 0.9975 & 0. & 0.\\ 0. & 0. & 0.973 & 0.\\ 0. & 0. & 0. & 0.7927\\ \end{bmatrix}`"

    dc : np.ndarray = np.array([0.2173,  0.2240, 0.2824, 0.2763]) * co2_to_C
    r"Transition vector from the emission to the boxes, by default :math:`\begin{bmatrix}0.2173\\0.224\\0.2824\\0.2763\\ \end{bmatrix} \times co2\_to\_C`"

    bc : np.ndarray = np.ones(4)
    r"Transition vector from the boxes to the atmospheric layer, by default :math:`\begin{bmatrix}1. & 1. & 1. & 1.\\ \end{bmatrix}`"

    initial_state : np.ndarray = np.array([139.1, 90.2, 29.2, 4.2]) 
    r"Initial content of the boxes, by default :math:`\begin{bmatrix}   139.1\\   90.2\\   29.2\\   4.2\\ \end{bmatrix}`"

    Ac5  =  exact_discretization(Ac - np.eye(4), np.expand_dims(dc,0).T, 5)[0]
    r"Exact transition matrix between the differents boxes for 5 years"

    dc5  =  exact_discretization(Ac - np.eye(4), np.expand_dims(dc,0).T, 5)[1]
    r"Exact transition vector from the emission to the boxes for 5 years"


    def five_years_cycle(self, emission : float, state : np.ndarray) -> np.ndarray:

        state = self.Ac5 @ state + self.dc5 * emission

        return state 

    def five_years_cycle_(self, emission : np.ndarray, state : np.ndarray) -> np.ndarray:
        r"""Function that calculate one cycle of carbon for a timestep of 5 years with the JOOS carbon cycle with zero order hold assumption. 

        Parameters
        ----------
            emission (np.ndarray): Emission over the five years in GtCO2/year

        Returns
        -------
            np.ndarray: Carbon in the boxes after 5 years GtC
        """

        for t in range(int(5 / self.timestep)):
            state = self.Ac @ state + self.dc * emission
        return state
        
    
    def atmospheric_carbon(self, state : np.ndarray) -> float:
        r"""Function that calculate the CO2 concentration in the atmosphere after 5 years from a initial state.

        Parameters
        ----------
            state (np.ndarray): Initial CO2 concentration in the 5 boxes

        Returns
        -------
            float: Carbon in the atmosphere after 5 years in GtC
        """

        return self.bc @ state + C_1750  # See code S. Dietz

    def five_years_atmospheric_carbon_old(self, emission : float, state : np.ndarray) -> float:
        r"""Function that calculate the CO2 concentration in the atmosphere after 5 years from a initial state.

        Parameters
        ----------
            emission (float): Emissions over the five years in GtCO2/year

        Returns
        -------
            float: Carbon in the atmosphere after 5 years in GtC
        """

        state = self.five_years_cycle(emission, state)
        return self.atmospheric_carbon(state) 

    def five_years_atmospheric_carbon(self, emission : float, state : np.ndarray) -> float:
        r"""Function that calculate the CO2 concentration in the atmosphere after 5 years from a initial state.

        Parameters
        ----------
            emission (float): Emissions over the five years in GtCO2/year

        Returns
        -------
            float: Carbon in the atmosphere after 5 years in GtC
        """
        bdc5 = self.bc @ self.dc5
        return self.bc @ self.Ac5 @ state + bdc5 * emission + C_1750

    def n_years_cycle(self, n : int, emission : np.ndarray) -> None:
        """Function which return the Carbon state in GtC after :math:`n` years for a given emission per years.

        Parameters
        ----------
        n : int
            Numbers of years
        emission : np.ndarray
            Emission over the five years in GtCO2/year
        """
        Acn, dcn  =  exact_discretization(self.Ac - np.eye(4), np.expand_dims(self.dc,0).T, n)
        state = Acn @ state + dcn * emission

        return state 



# class Carbon_FAIR(Linear_Carbon_Model) :

#     """Carbon model of "Joos et al." 2013. Ref Simon Dietz 2020
#     """

#     def __init__(self) -> None:
#         self.timestep = 1
#         self.Ac = np.array([[1, 0,      0,      0],
#                             [0, 0.9975, 0,      0],
#                             [0, 0,      0.9730, 0],
#                             [0, 0,      0,      0.7927]])
#         self.A = self.Ac - np.eye(4)
#         self.Ac5 = np.array([[1.        , 0.        , 0.        , 0.        ],
#                             [0.        , 0.9875778 , 0.        , 0.        ],
#                             [0.        , 0.        , 0.87371591, 0.        ],
#                             [0.        , 0.        , 0.        , 0.35469394]])

#         self.dc = np.array([0.2173,  0.2240, 0.2824, 0.2763]) * co2_to_C * self.timestep
#         self.dc5 = np.array([1.0865    , 1.11302908, 1.32083802, 0.86009679]) * co2_to_C
                              
#         self.bc = np.ones_like(self.dc) 
#         self.initial_state = np.array([139.1, 90.2, 29.2, 4.2])
#         self.name = 'Joos-al'


#     def reset(self) -> None:
#         self.initial_state = np.array([139.1, 90.2, 29.2, 4.2])

#         # Exact

#     def five_years_cycle(self, emission : float, state : np.ndarray) -> np.ndarray:
#         """Function that calculate one cycle of carbon for a timestep of 5 years with the JOOS carbon cycle.

#         Parameters
#         ----------
#             emission (np.ndarray): Emission over the five years in GtCO2/year

#         Returns
#         -------
#             np.ndarray: Carbon in the boxes after 5 years GtC
#         """

#         state = self.Ac5 @ state + self.dc5 * emission

#         return state 

#     def n_years_cycle_(self, n : int, emission : np.ndarray, state : np.ndarray) -> np.ndarray:
#         """Function that calculate one cycle of carbon for a timestep of 5 years with the JOOS carbon cycle.

#         Parameters
#         ----------
#             emission (np.ndarray): Emission over the n years in GtCO2/year

#         Returns
#         -------
#             np.ndarray: Carbon in the boxes after n years GtC
#         """

#         for i in range(0, n):
#             state = self.Ac @ state + self.dc * emission
#         return state
        
    
#     def atmospheric_carbon(self, state : np.ndarray) -> float:
#         """Function that calculate the CO2 concentration in the atmosphere after 5 years from a initial state.

#         Parameters
#         ----------
#             state (np.ndarray): Initial CO2 concentration in the 5 boxes

#         Returns
#         -------
#             float: Carbon in the atmosphere after 5 years in GtC
#         """

#         return self.bc @ state + C_1750 

#     def five_years_atmospheric_carbon(self, emission : float, state : np.ndarray) -> float:
#         """Function that calculate the CO2 concentration in the atmosphere after 5 years from a initial state.

#         Parameters
#         ----------
#             emission (float): Emissions over the five years in GtCO2/year

#         Returns
#         -------
#             float: Carbon in the atmosphere after 5 years in GtC
#         """

#         state = self.five_years_cycle(emission, state)
#         return self.atmospheric_carbon(state)


class Carbon_GHKT14(Linear_Carbon_Model) :
    """Non Used


    """
    def __init__(self) -> None:
        self.phi = 0.0228
        self.thetaL = 0.2
        self.theta0 = 0.393 # the proportion of atmospheric carbon in the transitory box that decays within the span of a unit of time
        self.Ac = np.diag(np.array([self.phi, 1 - self.phi]))
        self.dc = np.array([self.thetaL,   self.theta0(1 - self.thetaL)])
        self.bc = np.ones_like(self.dc)
        self.timestep = 10

class Carbon_GL18(Linear_Carbon_Model) :
    """Non used

    """

    def __init__(self) -> None:
        self.Ac = np.array([[0.6975,    0.2131, 0.029 ],
                            [0.1961,    0.7869, 0     ],
                            [0.1063,    0,      0.9706]])
        self.dc = np.array([0.914,   0.0744,  0.0447])
        self.bc = np.array([0.914, 0, 0])
        self.timestep = 10
