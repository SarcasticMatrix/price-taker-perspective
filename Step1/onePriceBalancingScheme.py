import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random


def onePriceBalancingScheme(
        scenarios: list,
        seed: int = 42 
) -> gp.Model:
    """
    Implement model for the Offering Strategy Under a One-Price Balancing Scheme

    Inputs:
    - scenarios (list of pd.dataframe): list of all the 250 scenarios, obtained from the function 'scenarios_selection_250' 
    
    Ouputs:
    - m (gp.Model): model optimise 
    """
    random.seed(seed)

    m = gp.Model("Offering Strategy Under a One-Price Balancing Scheme")

    ### Forecast inputs and model parameters
    P_nominal = 200
    pi = 1 / len(scenarios)
    price_DA = np.array([scenarios[i]['Price DA'].values for i in range(len(scenarios))])
    price_DA = np.transpose(price_DA)
    wind_production = np.array([scenarios[i]['Wind production'].values for i in range(len(scenarios))])
    wind_production = np.transpose(wind_production)
    imbalance = np.array([scenarios[i]['Power system need'].values for i in range(len(scenarios))])
    imbalance = np.transpose(imbalance)

    # --> TO DO: generate scenarios for p_{t,w}^real <--
    power_need += power_need * imbalance * np.random.normal(size=(24,len(scenarios)))
    power_need -= power_need * imbalance * np.random.normal(size=(24,len(scenarios)))

    ### Variables
    production_DA = m.addMVar(
        shape=(24,), lb=0, ub=P_nominal, name="Power generation for 24 hours", vtype=GRB.CONTINUOUS
    )
    delta = m.addMVar(
        shape=(24,len(scenarios)), name="Forecast deviation for 24 hours for 250 scenarios", vtype=GRB.CONTINUOUS
    )
    delta_up = m.addMVar(
        shape=(24,len(scenarios)), lb=0, name="Positive forecast deviation for 24 hours for 250 scenarios", vtype=GRB.CONTINUOUS
    )
    delta_down = m.addMVar(
        shape=(24,len(scenarios)), lb=0, name="Negative forecast deviation for 24 hours for 250 scenarios", vtype=GRB.CONTINUOUS
    )

    ### Objective function
    objective = m.setObjective(
        sum( 
            sum(
                pi * (price_DA[t,w] * production_DA[t] + 0.9 * price_DA[t,w] * delta_up[t,w] - 1.2 * price_DA[t,w] * delta_down[t,w])
                for w in range(len(scenarios))
                )
            for t in range(24)), GRB.MAXIMIZE)    

    ### Constraints
    m.addConstrs(
        (delta[t,w] == power_need[t,w] - production_DA[t]
        for t in range(24) for w in range(len(scenarios))),
        name="Delta definition with p_{t,w}^real and p_t^DA"
    )

    m.addConstrs(
        (delta[t,w] == delta_up[t,w] - delta_down[t,w] 
        for t in range(24) for w in range(len(scenarios))), 
        name="Delta definition with Delta_up and Delta_down"
    )

    m.optimize()

    return m
