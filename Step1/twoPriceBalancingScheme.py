import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import json

def export_results(model: gp.Model):
    """
    Export the retults to result.json
    """

    values = model.getAttr("X", model.getVars())
    names = model.getAttr("VarName", model.getVars())
    name_to_value = {name: value for name, value in zip(names, values)}

    with open("result.json", "w") as f:
        json.dump(name_to_value, f, indent=1)

def twoPriceBalancingScheme(
    scenarios: list, seed: int = 42, export: bool = False
) -> gp.Model:
    """
    Implement model for the Offering Strategy Under a Two-Price Balancing Scheme

    Inputs:
    - scenarios (list of pd.dataframe): list of all the 250 scenarios, obtained from the function 'scenarios_selection_250' 
    
    Ouputs:
    - m (gp.Model): model optimise 
    """
    random.seed(seed)

    m = gp.Model("Offering Strategy Under a Two-Price Balancing Scheme")

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
        shape=(24,len(scenarios)), lb=-np.inf, name="Forecast deviation for 24 hours for 250 scenarios", vtype=GRB.CONTINUOUS
    )
    delta_up = m.addMVar(
        shape=(24,len(scenarios)), lb=0, name="Upward forecast deviation for 24 hours for 250 scenarios", vtype=GRB.CONTINUOUS
    )
    delta_down = m.addMVar(
        shape=(24,len(scenarios)), lb=0, name="Downward forecast deviation for 24 hours for 250 scenarios", vtype=GRB.CONTINUOUS
    )
    binary = m.addMVar(
        shape=(24,len(scenarios)),vtype=GRB.BINARY, name="binary"
    )

    ### Objective function
    objective = m.setObjective(
        sum( 
            sum(
                pi*(
                    price_DA[t,w] * production_DA[t]
                    + (1 - power_needed[t,w]) * price_DA[t,w] * (0.9 * delta_up[t,w] - delta_down[t,w])
                    + power_needed[t,w] * price_DA[t,w] * (delta_up[t,w] - 1.2 * delta_down[t,w])
                ) 
                for w in range(len(scenarios))
            )
            for t in range(24)
        ), 
        GRB.MAXIMIZE
    )    

    ### Constraints
    m.addConstrs(
        (delta[t,w] == wind_production[t,w] - production_DA[t]
        for t in range(24) for w in range(len(scenarios))),
        name="Delta definition with p_{t,w}^real and p_t^DA"
    )
    m.addConstrs(
        (
            delta[t,w] == delta_up[t,w] - delta_down[t,w]
            for t in range(24) for w in range(len(scenarios))
        ),
        name="Delta definition with delta_up and delta_down"
    )
    m.addConstrs(
        (
            delta_up[t,w] <= 200 * binary[t,w]
            for t in range(24) for w in range(len(scenarios))
        ),
        name="Delta_up boundary"
    )
    m.addConstrs(
        (
            delta_up[t,w] <= 200 * (1 - binary[t,w])
            for t in range(24) for w in range(len(scenarios))
        ),
        name="Delta_down boundary"
    )
    
    m.optimize()
    if m.status == 2 and export:
        export_results(m)
    elif m.status != 2 and export:
        print("Model have not converged - impossible to export results to json")
    return m


import matplotlib.pyplot as plt


def conduct_analysis(scenarios: list, m: gp.Model):
    """
    Analyzes the results of the optimization model for the offering strategy under a one-price balancing scheme.

    Inputs:
    - m (gp.Model): Optimized model returned by onePriceBalancingScheme()

    Returns:
    - expected_profit (float): Expected profit
    """

    # production_DA = m.getAttr("X", m.getVars())[0:24]
    production_DA = [var.X for var in m.getVars() if "Power generation for 24 hours" in var.VarName]
    production_DA = np.hstack((production_DA, production_DA[-1]))
    price_DA = np.array(
        [scenarios[i]["Price DA"].values for i in range(len(scenarios))]
    )
    price_DA = np.transpose(price_DA)

    ### Delta_{t,w}
    # delta = m.getAttr("X", m.getVars())[24:24 + len(scenarios)*24]
    delta = [var.X for var in m.getVars() if "Forecast deviation for 24 hours for 250 scenarios" in var.VarName]
    delta = np.array(delta).reshape(24, 250).T
    delta = np.sort(delta, 0)
    delta_max = delta[-1, :]
    delta_max = np.hstack((delta_max, delta_max[-1]))
    delta_mean = delta.mean(axis=0)
    delta_mean = np.hstack((delta_mean, delta_mean[-1]))
    delta_min = delta[0, :]
    delta_min = np.hstack((delta_min, delta_min[-1]))

    P_nominal = 200
    wind_production_forecast = P_nominal * np.array(
        [scenarios[i]["Wind production"].values for i in range(len(scenarios))]
    )
    wind_production_forecast = np.sort(wind_production_forecast, 0)
    wind_max = wind_production_forecast[-1, :]
    wind_max = np.hstack((wind_max, wind_max[-1]))
    wind_mean = wind_production_forecast.mean(axis=0)
    wind_mean = np.hstack((wind_mean, wind_mean[-1]))
    wind_min = wind_production_forecast[0, :]
    wind_min = np.hstack((wind_min, wind_min[-1]))

    power_system_need = np.array(
        [scenarios[i]["Power system need"].values for i in range(len(scenarios))]
    )
    # power_system_need = np.transpose(power_system_need)
    power_system_need = np.sort(power_system_need, 0)
    power_system_need_max = power_system_need[-1, :]
    power_system_need_max = np.hstack((power_system_need_max, power_system_need_max[-1]))
    power_system_need_mean = power_system_need.mean(axis=0)
    power_system_need_mean = np.hstack((power_system_need_mean, power_system_need_mean[-1]))
    power_system_need_min = power_system_need[0, :]
    power_system_need_min = np.hstack((power_system_need_min, power_system_need_min[-1]))
    
    time = [i for i in range(25)]   

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    ### p_{t}^{DA} 
    ax1.step(time, wind_max, color="purple", linestyle="dotted", where="post", linewidth=1)
    ax1.step(time, wind_mean, color="purple", linestyle="solid", where="post", linewidth=1)
    ax1.step(time, wind_min, color="purple", linestyle="dashed", where="post", linewidth=1)
    ax1.step(np.nan,np.nan,color="purple",linestyle="solid",label="Power availability",where="post",linewidth=0.7,)
    ax1.step(time, production_DA, label=r"$p_{t}^{DA}$", where="post", color="red")
    ax1.set_title(r"DA offered production $p_t^{DA}$ and wind power forecast $p_{t,w}^{real}$")
    ax1.set_ylabel("Power [MW]")
    ax1.grid(visible=True,which="both",linestyle="--",color="gray",linewidth=0.5,alpha=0.8,)
    ax1.grid(which="minor", alpha=0.5)
    ax1.legend()

    ### \Delta_{t,w}
    ax2.step(time, delta_max, color="blue", linestyle="dotted", where="post", linewidth=1)
    ax2.step(time, delta_mean, color="blue", linestyle="solid", where="post", linewidth=1)
    ax2.step(time, delta_min, color="blue", linestyle="dashed", where="post", linewidth=1)
    ax2.step(np.nan,np.nan,color="blue",linestyle="solid",label=r"$\Delta_{t,w}$",linewidth=0.7,)
    ax2.set_title(r"Planned power deviation from forecasts $\Delta_{t,w}$")
    ax2.set_ylabel("Power [MW]")
    ax2.grid(visible=True,which="both",linestyle="--",color="gray",linewidth=0.5,alpha=0.8,)
    ax2.grid(which="minor", alpha=0.5)
    ax2.legend()

    ### x_{t,w}^B
    ax3.step(time, power_system_need_max, color="green", linestyle="dotted", where="post", linewidth=1)
    ax3.step(time, power_system_need_mean, color="green", linestyle="solid", where="post", linewidth=1)
    ax3.step(time, power_system_need_min, color="green", linestyle="dashed", where="post", linewidth=1)
    ax3.step(np.nan,np.nan,color="blue",linestyle="solid",label=r"$x_{t,w}^B$",linewidth=0.7,)
    ax3.set_title(r"Power system need $x_{t,w}^B$")
    ax3.set_xlabel("Hours")
    ax3.set_yticks([0,1],['Excess', 'Deficit'])
    ax3.grid(visible=True,which="both",linestyle="--",color="gray",linewidth=0.5,alpha=0.8,)
    ax3.grid(which="minor", alpha=0.5)
    ax3.legend()    

    # plt.suptitle("Hourly offered production in the day-ahead market", fontweight='bold')
    # plt.tight_layout()
    plt.xticks(time, [f"H{i}" for i in range(24)] + ["H0"])
    plt.show()

   # Profit plot

    profits = []
    for w in range(len(scenarios)):
        profit_w = sum(price_DA[t, w] * production_DA[t] for t in range(24))
        profits.append(profit_w)
    profits = np.array(profits)
    expected_profit = np.mean(profits)
    standard_deviation = np.std(profits, ddof=1)

    plt.figure()
    plt.hist(profits/10**3, bins=20, edgecolor='None', color='red', alpha=0.3)
    plt.hist(profits/10**3, bins=20, edgecolor="black", facecolor='None')
    plt.axvline(expected_profit/10**3, color='purple', label='Expected profit')
    plt.title(f"Profit Distribution Over Scenarios - Expected profit {round(expected_profit)} and its standard deviation {round(standard_deviation)}")
    plt.xlabel("Profit (k€)")
    plt.minorticks_on()
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    return expected_profit