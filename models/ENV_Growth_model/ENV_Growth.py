import numpy as np

class economic:

    def __init__(self,
                 g,
                 population,
                 population_employment,
                 h,
                 unr,
                 tfp,
                 tfp_frontier,
                 e_0,
                 rho_0,
                 rho_open,
                 openness,
                 open_world,
                 fe,
                 pmr,
                 global_pmr,
                 a_pmr,
                 capital,
                 capital_depreciation,
                 investment,
                 y_growth,
                 g_human,
                 g_population,
                 alpha_k_y,
                 delta,
                 emission_intensity,
                 mix,
                 elasticity : float = 0.31,
                 gamma_unr : float = 0.99,
                 unr_frontier : float = 0.02,
                 gamma_mpc : float = 0.985,
                 mpc_frontier : float = 0.1,
                 gamma_i :float = 0.98,
                 gamma_fe : float = 0.975,
                 fe_frontier : float = 0.105,
                 ) -> None:


    # Global 

        self.elasticity = elasticity

        #: global frontier growth rate
        self.g = g       


    # Population (L)

        self.population = population
        self.population_employment = population_employment

        # Human Capital 
        self.h = h

        # Unemployement
        self.unr = unr # rate
        self.gamma_unr = gamma_unr
        self.unr_frontier = unr_frontier


    # Total Factor Productivity (TFP)

        self.tfp = tfp 

        # TFP Frontier
        self.tfp_frontier = tfp_frontier
        self.e_0 = e_0

        # Openness
        self.rho_0 = rho_0
        self.rho_open = rho_open
        self.open = openness
        self.open_world = open_world

        # Fixed Effect (fe)
        self.fe = fe
        self.gamma_fe = gamma_fe
        self.fe_frontier = fe_frontier

        # Product market regulation (PMR)
        self.pmr = pmr
        self.global_pmr = global_pmr
        self.a_pmr = a_pmr 


    # Capital (K)

        self.capital = capital
        self.capital_depreciation = capital_depreciation

        # Investment

        self.investment = investment
        self.gamma_i = gamma_i
        self.y_growth = y_growth  # a way to get reed of the Y_t/ Y_{t-1} in the investment making it exogeneous

        # Return to capital (MCP)

        self.gamma_mpc = gamma_mpc
        self.mpc_frontier = mpc_frontier

        # Growth rate labor and education

        self.g_human = g_human
        self.g_population = g_population

        # Long-run Capital Output (k_y)

        self.alpha_k_y = alpha_k_y

        # Long-run target

        self.delta = delta



    # Energy and Emission (E)

        self.emission_intensity = emission_intensity
        self.mix = mix

        


    def gdp(self, previous_capital : np.ndarray, population : np.ndarray, population_employement : np.ndarray, tfp : np.ndarray):
        return tfp * previous_capital**self.elasticity * (population_employement * population)**(1 - self.elasticity)
    

# Labor (L)

    def labor_dynamic(self, unr : np.ndarray, aggregated_pop : np.ndarray):
        return (1-unr) *  aggregated_pop

    def unr_dynamic(self, initial_unr : float,  t : int):
        unr = []
        u = initial_unr
        for _ in range(t):
            unr.append(u)
            u = self.gamma_unr * u + (1  - self.gamma_unr) * self.unr_frontier

    def gloabal_labor(self, inital_unr : float, aggregated_pop : np.ndarray, t):
        unr = self.unr_dynamic(inital_unr, t)
        labor = self.labor_dynamic(unr, aggregated_pop)
        return labor
    

# TFP

    def fe_dynamic(self, initial_fe : float, t : int):
        fe = []
        f = initial_fe
        for _ in range(t):
            fe.append(f)
            f = self.gamma_fe * f + (1 - self.gamma_fe) * self.fe_frontier
        return np.array(fe)

    def rho_tfp_dynamic(self):
        # rho_tfp  = []
        # for idx in range(t):
        #     rho_tfp.append(self.rho_0 + self.rho_open[idx] * (self.open[idx] - self.open_world))/(1 + self.rho_0 +  self.rho_open[idx] * (self.open[idx] - self.open_world))
        return (self.rho_0 + self.rho_open * (self.open - self.open_world))/(1 + self.rho_0 +  self.rho_open * (self.open - self.open_world))
    
    def tfp_frontier_dynamic(self, fe : np.ndarray, t : int, t_0 : int):
        return np.exp( self.e_0 + fe + self.g * (np.arange(t) - t_0) + self.a_pmr * (self.pmr - self.global_pmr))

    def tfp_dynamic(self, initial_tfp : float, tfp_frontier : np.ndarray, rho_tfp : np.ndarray, t : int):
        tfp = []
        a = initial_tfp
        for idx in range(t):
            tfp.append(a)
            a = a * (tfp_frontier/ a) ** rho_tfp[idx]
        return np.array(tfp)
    
    def global_tfp(self, intial_tfp : float, intial_fe : float, t : int, t_0 : int):
        fe = self.fe_dynamic(intial_fe, t)
        rho_tfp = self.rho_tfp_dynamic(t)
        tfp_frontier = self.tfp_frontier_dynamic(fe, t, t_0) 
        tfp = self.tfp_dynamic(intial_tfp, tfp_frontier, rho_tfp, t)
        return tfp

# Capital 

    def mpc_dynamic(self, initial_mpc : float, t : int): 
        mpc = []
        m = initial_mpc
        for _ in range(t):
            mpc.append(m)
            m = self.gamma_mpc * m + (1 - self.gamma_mpc) * self.mpc_frontier
        return np.array(mpc)
    
    # long-run capital output

    def k_y_dynamic(self, mpc : np.ndarray):
        return self.alpha_k_y / mpc
    
    def g_y_dynamic(self,g : float,  g_population : np.ndarray, g_human : np.ndarray):
        return (1 + g) * (1 + g_population) * (1 + g_human) -1


    def i_y_dynamic(self, k_y : np.ndarray, g_y : np.ndarray):
        return (g_y + self.delta ) * k_y
    
    
    def i_dynamic(self, initial_investment : float, i_y : np.ndarray, y_growth : np.ndarray, t : int):
        investment = []
        i = initial_investment
        for idx in range(t):
            investment.append(i)
            i = self.gamma_i * y_growth * i + (1 - self.gamma_i) * i_y[idx]
        return np.array(investment)


    def capital_dynamic(self, initial_capital : float, investment : np.ndarray, t : int):
        capital = []
        k = initial_capital
        for idx in range(t):
            k = (1 - self.capital_depreciation) * k + investment[idx]
            capital.append(k)
        return np.array(capital)
    
    def global_capital(self, initial_mpc, initial_capital, initial_investment, t : int):
        mpc = self.mpc_dynamic(initial_mpc, t)
        k_y = self.k_y_dynamic(mpc)
        g_y = self.g_y_dynamic(self.g, self.g_population, self.g_human)
        i_y = self.i_y_dynamic(k_y, g_y)
        investment = self.i_dynamic(initial_investment, i_y, self.y_growth, t)
        capital = self.capital_dynamic(initial_capital, investment, t)
        return capital
    

    # Energy / CO2 comsumption

    # def 

    

        
        