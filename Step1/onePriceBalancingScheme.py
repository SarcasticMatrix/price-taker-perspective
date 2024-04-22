import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random


def export_results(model:gp.Model):
    """
    Export the retults to result.json
    """

    values = model.getAttr("X", model.getVars())
    names = model.getAttr("VarName", model.getVars())
    name_to_value = {
        name: value for name, value in zip(names,values)
    }
    import json
    with open('result.json', 'w') as f:
        json.dump(name_to_value, f, indent=1)

def onePriceBalancingScheme(
        scenarios: list,
        seed: int = 42,
        export: bool = False
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

    if m.status == 2 and export:
        export_results(m)
    elif m.status != 2 and export:
        print("Model have not converged - impossible to export results to json")
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

    production_DA = m.getAttr("X", m.getVars())[0:24]
    price_DA = np.array([scenarios[i]['Price DA'].values for i in range(len(scenarios))])
    price_DA = np.transpose(price_DA)

    delta = m.getAttr("X", m.getVars())[24:]
    delta = np.array(delta).reshape(24,250).T
    delta = np.sort(delta, 0)

    profits = []
    for w in range(len(scenarios)):
        profit_w = sum(price_DA[t, w] * production_DA[t] for t in range(24))
        profits.append(profit_w)

    expected_profit = np.mean(profits)

    P_nominal = 200
    wind_production_forecast = P_nominal * np.array([scenarios[i]['Wind production'].values for i in range(len(scenarios))])
    wind_production_forecast = np.sort(wind_production_forecast, 0)

    time = [i for i in range(25)]

    wind_max = wind_production_forecast[-1, :]
    wind_max = np.hstack((wind_max, wind_max[-1]))
    wind_mean = wind_production_forecast.mean(axis=0)
    wind_mean = np.hstack((wind_mean, wind_mean[-1]))
    wind_min = wind_production_forecast[0, :]
    wind_min = np.hstack((wind_min, wind_min[-1]))

    delta_max = delta[-1, :]
    delta_max = np.hstack((delta_max, delta_max[-1]))
    delta_mean = delta.mean(axis=0)
    delta_mean = np.hstack((delta_mean, delta_mean[-1]))
    delta_min = delta[0, :]
    delta_min = np.hstack((delta_min, delta_min[-1]))

    production_DA = np.hstack((production_DA, production_DA[-1]))


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.step(time, wind_max, color='purple', linestyle='dotted', where='post', linewidth=1)
    ax1.step(time, wind_mean, color='purple', linestyle='solid', where='post', linewidth=1)
    ax1.step(time, wind_min, color='purple', linestyle='dashed', where='post', linewidth=1)
    ax1.step(np.nan, np.nan, color='purple', linestyle='solid', label='Power availability', where='post', linewidth=0.7)

    ax1.step(time, production_DA, label=r'$p_{t}^{DA}$', where='post', color='red')

    ax1.set_title(r"DA offered production $p_t^{DA}$ and wind power forecast $p_{t,w}^{real}$")
    ax1.set_ylabel('Power [MW]')
    ax1.grid(visible=True, which='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.8)
    ax1.grid(which='minor', alpha=0.5)
    ax1.legend()

    ax2.step(time, delta_max, color='blue', linestyle='dotted', where='post', linewidth=1)
    ax2.step(time, delta_mean, color='blue', linestyle='solid', where='post', linewidth=1)
    ax2.step(time, delta_min, color='blue', linestyle='dashed', where='post', linewidth=1)
    ax2.step(np.nan, np.nan, color='blue', linestyle='solid', label='Planned imbalance', linewidth=0.7)

    ax2.set_title(r"Planned power deviation from forecasts $\Delta_{t,w}$")
    ax2.set_xlabel('Hours')
    ax2.set_ylabel('Power [MW]')
    ax2.grid(visible=True, which='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.8)
    ax2.grid(which='minor', alpha=0.5)
    ax2.legend()

    # plt.suptitle("Hourly offered production in the day-ahead market", fontweight='bold')
    plt.tight_layout()
    plt.xticks(time, [f"H{i}" for i in range(24)] + ['H0'])
    plt.show()


    plt.figure()
    plt.hist(profits, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Profit Distribution Over Scenarios - Expected profit {expected_profit}')
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    return expected_profit