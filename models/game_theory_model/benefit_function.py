import numpy as np
import pandas as pd

def polynome(coefficients : np.ndarray) -> callable :
    """Polynomial benefit function.

    Parameters
    ----------
    coefficients : np.ndarray
        Coeffiecients of the polynomial.

    Returns
    -------
    callable
        Benefit function.
    """
    def benef(x):
        return np.polyval(np.flip(coefficients),x)
    return benef

def benefit_affine(GDP_max : float, e_max :float, percentage_green : float = 0 ) : 
    r"""Basic affine benefit funtion. This function attain its maximum at maximum emsission. 
    

    Parameters
    ----------
    GDP_max : float
        Maximum GDP attainable.
    e_max : float
        Maximum emission.
    percentage_green : float, optional
        Percentage of the benefit function independant of CO2, by default 0

    Returns
    -------
    callable
        Benefit function taking into argument CO2 emissions.

    Notes
    -----

    The function is given by

    .. math:: 

        B(X) = \frac{GDP_{max} (1 - \%_{green})}{e_{max}} X + GDP_{max} \%_{green}
    """
    def benef(x) :
        return GDP_max * (1-percentage_green) * x /e_max + GDP_max*percentage_green
    return benef

def benefit_quadratic_concave(GDP_max : float, e_max : float, percentage_green : float =0) -> callable :
    r"""Basic concave benefit funtion. This function attain its maximum at maximum emsission. 
    

    Parameters
    ----------
    GDP_max : float
        Maximum GDP attainable.
    e_max : float
        Maximum emission.
    percentage_green : float, optional
        Percentage of the benefit function independant of CO2, by default 0

    Returns
    -------
    callable
        Benefit function taking into argument CO2 emissions.

    Notes
    -----

    The function is given by

    .. math:: 

        B(X) = -\frac{GDP_{max} (1 - \%_{green})}{e_{max}^2} X^2 + 2 \frac{GDP_{max} (1 - \%_{green})}{e_{max}} X + GDP_{max} \%_{green}
    """
    benef_coef = np.array([GDP_max * percentage_green , 2 * (GDP_max - GDP_max * percentage_green) /e_max, -(GDP_max - GDP_max * percentage_green) /e_max**2 ])
    def benef(x):
        return np.polyval(np.flip(benef_coef),x)
    return benef


def benefit_quadratic_convex(GDP_max : float, e_max : float, percentage_green : float =0) -> callable :
    r"""Basic convex benefit funtion. This function attain its maximum at maximum emsission. 
    

    Parameters
    ----------
    GDP_max : float
        Maximum GDP attainable.
    e_max : float
        Maximum emission.
    percentage_green : float, optional
        Percentage of the benefit function independant of CO2, by default 0

    Returns
    -------
    callable
        Benefit function taking into argument CO2 emissions.

    Notes
    -----

    The function is given by

    .. math:: 

        B(X) = \frac{GDP_{max} (1 - \%_{green})}{e_{max}^2} X^2 + GDP_{max} \%_{green}
    """
    benef_coef = np.array([GDP_max * percentage_green , 0, (GDP_max - GDP_max * percentage_green) /e_max**2 ])
    def benef(x):
        return np.polyval(np.flip(benef_coef),x)
    return benef

def benefit_quadratic_concave_with_green(GDP_max : float, e_max : float, coef_green : np.ndarray = 0.5, percentage_green : float =0) -> callable :
    benef_coef = np.array([GDP_max * percentage_green , 2 * (GDP_max - GDP_max * percentage_green) /e_max, -(GDP_max - GDP_max * percentage_green) /e_max**2 ])
    def benef(x):
        return np.polyval(np.flip(benef_coef),x) + (1-x)* coef_green
    return benef
    
def benefit_quadratic_convex_with_percentage_green(GDP_max : float, e_max : float, percentage_green : float =0) -> callable :
    
    benef_coef = np.array([GDP_max * percentage_green , 0, (GDP_max - GDP_max * percentage_green) /e_max**2 ])
    def benef(x):
        return np.polyval(np.flip(benef_coef),x)
    return benef


def polynome_fitted(data : pd.DataFrame, degree) -> callable :
    pass

def benefice_exponential(GDP_max : float, e_max : float, power : float = 0.3) : 
    def benef(x) :
        return ((1-np.exp(-(x/e_max)) )/(1-np.exp(-1)))**power * GDP_max
    return benef

def benefit_sigm(GDP_max : float, e_max :float, percentage_green =0 , power : float = 10) : 
    r"""Basic sigmoidal benefit funtion. This function attain its maximum at maximum emsission. 
    
    Parameters
    ----------
    GDP_max : float
        Maximum GDP attainable.
    e_max : float
        Maximum emission.
    percentage_green : int, optional
        Percentage of the benefit function independant of CO2, by default 0
    power : float, optional
        Power p of the sigmoid, by default 10
    

    Returns
    -------
    callable
            Benefit function taking into argument CO2 emissions.

    Notes
    -----

    The function is given by

    .. math:: 

        B(X) = (\frac{1 - e^{-X}}{1 - e^{e_{max}}})^{p} GDP_{max} (1 - \%_{green}) +  GDP_{max} \%_{green}

    """
    
    def benef(x) :
        return ( (1-np.exp(-x)) / (1-np.exp(-e_max)) )**power * GDP_max * (1 - percentage_green) + GDP_max * percentage_green
    return benef




# def polynome(coefficients : np.ndarray) -> callable :

#     benef = np.polynomial.Polynomial(coefficients)
#     return benef

# def benefit_quadratic_concave_with_percentage_green(GDP_max : float, e_max : float, percentage_green : float ) -> callable :
#     benef_coef = np.array([GDP_max * percentage_green , 2 * (GDP_max - GDP_max * percentage_green) /e_max, -(GDP_max - GDP_max * percentage_green) /e_max**2 ])

#     return np.polynomial.Polynomial(benef_coef)

# def benefit_quadratic_convex_with_percentage_green(GDP_max : float, e_max : float, percentage_green : float ) -> callable :
#     benef_coef = np.array([GDP_max * percentage_green , 0, (GDP_max - GDP_max * percentage_green) /e_max**2 ])

#     return np.polynomial.Polynomial(benef_coef)


# def polynome_fitted(data : pd.DataFrame, degree) -> callable :
#     pass

# def benefice_exponential(GDP_max : float, e_max : float, power : float = 0.3) : 
#     def benef(x) :
#         return ((1-np.exp(-(x/e_max)) )/(1-np.exp(-1)))**power * GDP_max
#     return benef

# def benefice_sigm(GDP_max : float, e_max :float, power : float = 10) : 
#     def benef(x) :
#         return ( (1-np.exp(-x)) / (1-np.exp(-e_max)) )**power * GDP_max
#     return benef

# def benefit_affine(GDP_max : float, e_max :float) : 
#     def benef(x) :
#         return GDP_max * x /e_max
#     return benef