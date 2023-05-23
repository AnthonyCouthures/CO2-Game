import numpy as np


def linear_benefit(investment, a, b):
    return a * investment + b

def quadratic_benefit(investment, a, b, c):
    return a * investment**2 + b * investment + c

def logarithmic_benefit(investment, a, b):
    return a * np.log(investment + 1) + b

def exponential_benefit(investment, a, b):
    return a * (1 - np.exp(-b * investment))
