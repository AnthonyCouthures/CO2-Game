r"""Temperature Dynamic structure.

The base model is discrete and linear.  

Linear models of temperature dynamic are mostly represented as flows of temperature between :math:`m' \in \mathbb{N}` different boxes. They are modelled as a system of :math:`m' \in \mathbb{N}` equations as the following 

.. math::

    \Theta(t) &= A_\Theta \Theta(t-1) + d_\Theta F(t) , \quad \forall t \in \mathbb{N} \\
    \Theta(0) &= \Theta_0

Where,

- :math:`\Theta(t)` is the `m'`-vector of Temperature in the boxes at time :math:`t`,

- :math:`F(t)` is the radiative forcing at time :math:`t`,

- :math:`A_\Theta` is the square matrix of constants that represent the temeprature flux between the boxes,

- :math:`d_\Theta` is the vector that share the radiative forcing to the boxes.

The number of boxes, the matrix :math:`A_\Theta` and the vector :math:`d_\Theta` differ from a model to another.  

The atmospheric temperature is given by,

.. math::

    \Theta_{AT}(t)=b_\Theta^{\mathrm{T}} \Theta(t), \quad \forall t \in \mathbb{N}

Where :math:`b_\Theta` is the vector that extract the atmospheric temperature :math:`\Theta_{AT}`.

"""



import numpy as np
from .usefull_method import exact_discretization


class Linear_Temperature_Dynamic :
    """Structure of a Linear Temperature Dynamic.
    """

    name : str = " "
    r"Name of the model"

    timestep : int = 1  
    r"Time-step of the model"

    At : np.ndarray = np.ones(1) 
    r"Transition matrix between the differents boxes"

    d : np.ndarray = np.ones(1) 
    r"Transition vector from the emission to the boxes"

    bt : np.ndarray = np.ones(1) 
    r"Transition vector from the boxes to the atmospheric layer"

    initial_state : np.ndarray = np.ones(1) 
    r"Initial content of the boxes"

    # def __init__(self,
    #             timestep : int = 1 ,
    #             At : np.ndarray = np.ones(1),
    #             d : np.ndarray = np.ones(1),
    #             bt : np.ndarray = np.ones(1),
    #             initial_state : np.ndarray = np.ones(1),
    #             name : str = '') -> None:


    #     self.timestep = timestep
    #     self.At = At
    #     self.d = d
    #     self.bt = bt
    #     self.initial_state = initial_state
    #     self.name = name

    
    def __copy__(self):
            return type(self)(self.At, self.d, self.bt, self.initial_state, self.name)

    def __deepcopy__(self, memo): 
        id_self = id(self)        
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)()
            memo[id_self] = _copy 
        return _copy



    def init_temp_dynamic(self, initial_state : np.ndarray) -> None:
        r"""Function which initialize the Temperature Dynamic from a given state.

        Parameters
        ----------
        initial_state : np.ndarray
            Intial state of the temperature dynamic.
        """
        self.initial_state = initial_state
    

    def five_years_cycle(self, forcing : float) -> np.ndarray:
        r"""Function which return the temperature state in °C after five years for a given radiative forcing.

        Parameters
        ----------
        forcing : float
            Forcing over the five years in w/m² per year.

        Returns
        -------
        state : np.ndarray
            Temperature in the boxes after 5 years in °C.
        """
        pass

    def atmospheric_temperature(self, state : np.ndarray) -> float:
        r"""Function which return the atmospheric temperature variation A EXPLIQUER PLUS from a state.

        Returns
        -------
        atmospheric_temperature : float
            Atmospheric temperature variation in °C.
        """

        return self.bt @ state

    def five_years_atmospheric_temperature(self, forcing : float, state : np.ndarray) -> float:
        r"""Function which return the atmospheric temperature variation A EXPLIQUER PLUS after five years from a state for a given emission.

        Parameters
        ----------
        forcing : float
            Forcing over the five years in w/m² per year.
        state : np.ndarray
            A state of the temperature dynamic.

        Returns
        -------
        atmospheric_temperature : float
            Atmospheric temperature variation in °C.
        """

        state = self.five_years_cycle(forcing, state)
        return self.atmospheric_temperature(state)



class Temp_DICE_2013R(Linear_Temperature_Dynamic) :
    r"""Temperature Dynamic from DICE-2013 model. 

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
    
    At : np.ndarray = np.array([[0.8630,  0.0086],
                            [0.025,   0.975]]) 
    r"Transition matrix between the differents boxes, by default :math:`\begin{bmatrix}   0.863 & 0.0086\\   0.025 & 0.975\\ \end{bmatrix}`"
    
    d : np.ndarray = np.array([0.1005,   0]) 
    r"Transition vector from the emission to the boxes, by default :math:`\begin{bmatrix}   0.1005\\   0.\\ \end{bmatrix}`"
    
    bt : np.ndarray = np.array([1,   0]) 
    r"Transition vector from the boxes to the atmospheric layer, by default :math:`\begin{bmatrix}   1 &   0 \end{bmatrix}`"
    
    initial_state : np.ndarray = np.array([1.01, 0.0068]) 
    r"Initial content of the boxes, by default :math:`\begin{bmatrix}   1.01\\   0.0068\\ \end{bmatrix}`"
    

    def five_years_cycle(self, forcing : float, state : np.ndarray) -> np.ndarray:

        state = self.At @ state + self.d * forcing 
        return state



class Temp_DICE_2016R(Linear_Temperature_Dynamic) :
    r"""Temperature Dynamic from DICE-2016R model. 

    References
    ----------

    .. [1] Nordhaus, William. « Estimates of the Social Cost of Carbon: Concepts and Results from the DICE-2013R Model and Alternative Approaches». Journal of the Association of Environmental and Resource Economists 1, nᵒ 1/2 (mars 2014): 273-312. https://doi.org/10.1086/676035.
    .. [2] Kellett, Christopher M., Steven R. Weller, Timm Faulwasser, Lars Grüne, et Willi Semmler. « Feedback, Dynamics, and Optimal Control in Climate Economics». Annual Reviews in Control 47 (1 janvier 2019): 7-20. https://doi.org/10.1016/j.arcontrol.2019.04.003.
    .. [3] Rezai, Armon, Simon Dietz, Frederick van der Ploeg, et Frank Venmans. « Are Economists Getting Climate Dynamics Right and Does It Matter? » Copernicus Meetings, 9 mars 2020. https://doi.org/10.5194/egusphere-egu2020-20039.


    """

    name : str = 'DICE-2016R'
    r"Name of the model"
    
    timestep : int = 5  
    r"Time-step of the model"
    
    At : np.ndarray = np.array([[0.8718,  0.0088],
                                [0.025,   0.975]]) 
    r"Transition matrix between the differents boxes, by default :math:`\begin{bmatrix}   0.8718 & 0.0088\\   0.025 & 0.975\\ \end{bmatrix}`"
    
    d : np.ndarray = np.array([0.1005,   0]) 
    r"Transition vector from the emission to the boxes, by default :math:`\begin{bmatrix}   0.1005\\   0.\\ \end{bmatrix}`"
    
    bt : np.ndarray = np.array([1,   0]) 
    r"Transition vector from the boxes to the atmospheric layer, by default :math:`\begin{bmatrix}   1 &   0 \end{bmatrix}`"
    
    initial_state : np.ndarray = np.array([1.01, 0.0068]) 
    r"Initial content of the boxes, by default :math:`\begin{bmatrix}   1.01\\   0.0068\\ \end{bmatrix}`"
    
    def five_years_cycle(self, forcing : float, state : np.ndarray) -> np.ndarray:

        state = self.At @ state + self.d * forcing 
        return state      
        

class Temp_Discret_Geoffroy(Linear_Temperature_Dynamic) :
    r"""Temperature Dynamic from Geoffroy et al. model. 

    Notes
    -----

    The temperature dynamic is given by the following EDO 

    .. math::

        &\dot{\Theta} = A_\Theta \Theta + d_\Theta F\\
        &\Theta_{AT}  = d_\Theta \Theta
    
    With the transition matrix and the vectors  given by

    .. math::

        A_\Theta = \begin{bmatrix}   -(\lambda + \gamma) /C & \gamma/C   \\ \gamma/C_0 &  - \gamma/C_0 \end{bmatrix}, \quad d_\Theta =  \begin{bmatrix} 1/C\\ 0 \end{bmatrix} \quad \text{ and } \quad b_\Theta = \begin{bmatrix} 1\\ 0 \end{bmatrix}


    References
    ----------

    .. [1] Geoffroy, O., D. Saint-Martin, D. J. L. Olivié, A. Voldoire, G. Bellon, et S. Tytéca. « Transient Climate Response in a Two-Layer Energy-Balance Model. Part I: Analytical Solution and Parameter Calibration Using CMIP5 AOGCM Experiments ». Journal of Climate 26, nᵒ 6 (15 mars 2013): 1841‑57. https://doi.org/10.1175/JCLI-D-12-00195.1.
    .. [2] Kellett, Christopher M., Steven R. Weller, Timm Faulwasser, Lars Grüne, et Willi Semmler. « Feedback, Dynamics, and Optimal Control in Climate Economics». Annual Reviews in Control 47 (1 janvier 2019): 7-20. https://doi.org/10.1016/j.arcontrol.2019.04.003.
    .. [3] Rezai, Armon, Simon Dietz, Frederick van der Ploeg, et Frank Venmans. « Are Economists Getting Climate Dynamics Right and Does It Matter? » Copernicus Meetings, 9 mars 2020. https://doi.org/10.5194/egusphere-egu2020-20039.


    """

    name : str = 'Geoffroy-et-al.'
    r"Name of the model"
    
    timestep : int = 1  
    r"Time-step of the model"

    C = 7.3
    r"The effective heat capacity of the upper/mixed ocean layer, by default 7.3"

    C0 = 106
    r"The effective heat capacity of the deep oceans, by default 106"

    lam = 1.3
    r"Parameter :math:`\lambda` of the model, by default 1.3"

    gam = 0.73
    r"Parameter :math:`\gamma` of the model, by default 0.73"
    
    At : np.ndarray = np.array([[-(lam + gam) / C, gam / C  ],
                                [gam / C0        , gam / C0]]) + np.eye(2)
    r"Transition matrix between the differents boxes"
    
    d : np.ndarray = np.array([(3.1 / 3.05) / C,   0]) 
    r"Transition vector from the radiative forcing to the boxes"
    
    bt : np.ndarray = np.array([1,   0]) 
    r"Transition vector from the boxes to the atmospheric layer, by default :math:`\begin{bmatrix}   1 &   0 \end{bmatrix}`"
    
    initial_state : np.ndarray = np.array([1.01, 0.0068]) 
    r"Initial content of the boxes, by default :math:`\begin{bmatrix}   1.01\\   0.0068\\ \end{bmatrix}`"
    
    At5  =  exact_discretization(At - np.eye(2), np.expand_dims(d,0).T, 5)[0]
    r"Exact transition matrix between the differents boxes for 5 years"

    d5  =  exact_discretization(At - np.eye(2), np.expand_dims(d,0).T, 5)[1]
    r"Exact transition vector from the radiative forcing to the boxes for 5 years"


    def five_years_cycle(self, forcing : float, state : np.ndarray) -> np.ndarray:

        state = self.At5 @ state + self.d5 * forcing

        return state 


        

# class Temp_Discret_FAIR20(Linear_Temperature_Dynamic) :
    
#     def __init__(self, forcing_2_time_CO2 : float = 3.45, ECS : float = 3.1, step : int = 1) -> None:
#         self.timestep = 5
#         self.C = 7.3
#         self.C0 = 106
#         self.lam = 1.3 # forcing_2_time_CO2/ECS
#         self.gam = 0.73

#         self.A = np.array([[-(self.lam + self.gam) / self.C, self.gam / self.C  ],
#                            [self.gam / self.C0             , -self.gam / self.C0]])
#         self.At = self.A + np.eye(2)   # Forward Euler for 1 years step

#         self.d = np.array([(3.1 / 3.05) / self.C,   0]) #  3.1/3.05 Simon Diez
#         self.d_ = np.array([[ (3.1 / 3.05) / self.C,   0]]).T #  3.1/3.05 Simon Diez

#         self.At5, self.d5 = exact_discretization(self.A, self.d_, time = self.timestep)

#         # self.At5= expm(self.A * self.timestep)  # Exact discretization over step 5 years
#         # self.d5 = np.linalg.inv(self.A) @ (self.At5 - np.eye(2)) @ self.d 

#         self.bt = np.array([1,   0])
#         self.name = 'Geoffroy-al'
#         self.initial_state = np.array([1.01, 0.0068]) # 2020 
#         # self.initial_state = np.array([1.01, 0.0068]) # 2015

#     def reset(self) -> None:
#         self.initial_state = np.array([1.01, 0.0068]) # 2020 
#         # self.initial_state = np.array([1.01, 0.0068]) # 2015



#     def five_years_cycle(self, forcing : float, state : np.ndarray) -> np.ndarray:
#         """Function that calculate one temperature dynamic for a timestep of 5 years with the Geoffroy and al. temperature dynamic.

#         Args:
#             forcing (float): Radiative forcing during the five years

#         Returns:
#             np.ndarray: temperature in the boxes after 5 years
#         """

#         state = self.At5 @ state + self.d5 * forcing

#         return state 
