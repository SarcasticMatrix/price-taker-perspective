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
    wind_production = P_nominal * np.array([scenarios[i]['Wind production'].values for i in range(len(scenarios))])
    wind_production = np.transpose(wind_production)
    power_needed = np.array([scenarios[i]['Power system need'].values for i in range(len(scenarios))])
    power_needed = np.transpose(power_needed)

    ### Variables
    production_DA = m.addMVar(
        shape=(24,), lb=0, ub=P_nominal, name="Power generation for 24 hours", vtype=GRB.CONTINUOUS
    )
    delta = m.addMVar(
        shape=(24,len(scenarios)), name="Forecast deviation for 24 hours for 250 scenarios", vtype=GRB.CONTINUOUS
    )

    ### Objective function
    objective = m.setObjective(
        sum( 
            sum(
                pi * (price_DA[t,w] * production_DA[t] +
                      (1 - power_needed[t,w]) * 0.9 * price_DA[t,w] * delta[t,w] + power_needed[t,w] * 1.2 * price_DA[t,w] * delta[t,w]
                      ) 
                for w in range(len(scenarios))
                )
            for t in range(24)), GRB.MAXIMIZE)    

    ### Constraints
    m.addConstrs(
        (delta[t,w] == wind_production[t,w] - production_DA[t]
        for t in range(24) for w in range(len(scenarios))),
        name="Delta definition with p_{t,w}^real and p_t^DA"
    )


    m.optimize()

    return m


import matplotlib.pyplot as plt

def conduct_analysis(
        scenarios: list,
        m: gp.Model
    ):
    """
    Analyzes the results of the optimization model for the offering strategy under a one-price balancing scheme.

    Inputs:
    - m (gp.Model): Optimized model returned by onePriceBalancingScheme()

    Returns:
    - expected_profit (float): Expected profit
    """

    ## TO DO: 
    ## - fix wind power forecast interval confidence 
    ## - afficher les ticks pour chaque heure

    production_DA = m.getAttr("X", m.getVars())[0:24]
    price_DA = np.array([scenarios[i]['Price DA'].values for i in range(len(scenarios))])
    price_DA = np.transpose(price_DA)

    profits = []
    for w in range(len(scenarios)):
        profit_w = sum(price_DA[t, w] * production_DA[t] for t in range(24))
        profits.append(profit_w)

    expected_profit = np.mean(profits)

    P_nominal = 200
    wind_production_forecast = P_nominal * np.array([scenarios[i]['Wind production'].values for i in range(len(scenarios))])
    wind_production_forecast = np.sort(wind_production_forecast, 0)

    time = [i for i in range(24)]
    plt.figure()

    plt.step(time, wind_production_forecast[0, :], color='green', label=r'Min power avalaible at time $t$', linestyle='--', where='post')
    plt.step(time, wind_production_forecast[-1, :], color='purple', label=r'Max power avalaible at time $t$', linestyle='--', where='post')
    plt.step(time, wind_production_forecast.mean(axis=0), color='blue', label=r'Mean power avalaible at time $t$', linestyle='--', where='post')

    # Nbr_scenarios = wind_production_forecast.shape[1]
    # cmap = plt.get_cmap('Blues') 
    # for i in range(Nbr_scenarios):
    #     if i < Nbr_scenarios - i:
    #         plt.fill_between(time, wind_production_forecast[i], wind_production_forecast[Nbr_scenarios - i - 1], color=cmap((i+1) / (Nbr_scenarios-1)), step='post')

    plt.step(time, production_DA, label=r'$p_{t}^{DA}$', where='post', color='red')

    plt.xlabel('Hours [h]')
    plt.ylabel('Power production [MW]')
    plt.title("Optimal hourly offered production in the day-ahead market")
    plt.grid(visible=True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(profits, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Profit Distribution Over Scenarios - Expected profit {expected_profit}')
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    return expected_profit