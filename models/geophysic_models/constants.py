


#: Conversion ratio of ppm of co2 to Gt of carbon
ppm_co2_to_c = 2.13

#: Conversion ratio of ppm of co2 to Gt of co2
ppm_co2_to_co2 = 7.8

#: Conversion ratio of Gt co2 into Gt of carbon equal to 12/44
co2_to_C = 12/44



CARBON_IN_CO2_EQUIVALENT_PPM_1750 = 278
"""Carbon Concentration in ppm at Equilibrium (i.e pre-industrial era) as CO2 equivalent

References
----------

.. [1] https://www.csiro.au/en/research/environmental-impacts/climate-change/state-of-the-climate/greenhouse-gases
.. [2] https://ourworldindata.org/explorers/climate-change?time=1750..2018&facet=none&hideControls=true&Metric=CO%E2%82%82+concentrations&Long-run+series%3F=true&country=~OWID_WRL"""

#: Carbon Concentration in GtC at Equilibrium (i.e pre-industrial era) 
C_1750 = CARBON_IN_CO2_EQUIVALENT_PPM_1750 * ppm_co2_to_c 

### Climate model parameters


F_2XCO2 = 3.7                   
"""Forcings of equilibrium CO2 doubling (Wm-2)

References
----------

.. [1] https://en.wikipedia.org/wiki/Climate_sensitivity"""