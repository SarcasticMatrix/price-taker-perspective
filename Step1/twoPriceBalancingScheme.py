import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import matplotlib.pyplot as plt

from Step1.analysis import export_results

def twoPriceBalancingScheme(
    scenarios: list, seed: int = 42, export: bool = False, optimise: bool = True
) -> gp.Model:
    """
    Implement model for the Offering Strategy Under a Two-Price Balancing Scheme

    Inputs:
    - scenarios (list of pd.dataframe): list of all the 250 scenarios, obtained from the function 'scenarios_selection_250'
    - seed (int): seed for random number generation
    - export (bool): flag to export results to JSON
    - optimise (bool): flag to indicate whether to perform optimization

    Outputs:
    - m (gp.Model): optimized model
    """

    # Create a new model
    m = gp.Model("Offering Strategy Under a Two-Price Balancing Scheme")

    ### Forecast inputs and model parameters
    P_nominal = 200
    nb_scenarios = len(scenarios)
    pi = 1 / nb_scenarios
    price_DA = np.array([scenarios[i]['Price DA'].values for i in range(nb_scenarios)])
    price_DA = np.transpose(price_DA)
    wind_production = P_nominal * np.array([scenarios[i]['Wind production'].values for i in range(nb_scenarios)])
    wind_production = np.transpose(wind_production)
    power_needed = np.array([scenarios[i]['Power system need'].values for i in range(nb_scenarios)])
    power_needed = np.transpose(power_needed)

    ### Variables
    # Define variables for power generation, forecast deviation, upward and downward forecast deviation, and binary variable
    production_DA = {
        t: m.addVar(lb=0, ub=P_nominal, name=f"DA power generation at time {t}.")
        for t in range(24)
    }

    delta = {
        t: {w: m.addVar(lb=-gp.GRB.INFINITY, name=f"Forecast deviation at time {t} for scenario {w}.") for w in range(nb_scenarios)}
        for t in range(24)
    }

    delta_up = {
        t: {
            w: m.addVar(lb=0, name=f"Upward forecast deviation at time {t} for scenario {w}.")
            for w in range(nb_scenarios)
        }
        for t in range(24)
    }

    delta_down = {
        t: {
            w: m.addVar(lb=0, name=f"Downward forecast deviation at time {t} for scenario {w}.")
            for w in range(nb_scenarios)
        }
        for t in range(24)
    }

    binary = {
        t: {
            w: m.addVar(lb=0, name=f"binary at time {t} for scenario {w}.", vtype=GRB.BINARY)
            for w in range(nb_scenarios)
        }
        for t in range(24)
    }
    
    ### Objective function
    # Set the objective function
    objective = gp.quicksum(
        gp.quicksum(
            pi
            * (
                price_DA[t, w] * production_DA[t]
                + power_needed[t, w] * price_DA[t, w] * (0.9 * delta_up[t][w] - delta_down[t][w])
                + (1 - power_needed[t, w]) * price_DA[t, w] * (delta_up[t][w] - 1.2 * delta_down[t][w])
            )
            for t in range(24)
        )
        for w in range(nb_scenarios)
    )
    m.setObjective(objective, gp.GRB.MAXIMIZE) 

    ### Constraints
    # Define constraints on forecast deviation
    delta_value = {
        t: {
            w: m.addConstr(
                delta[t][w],
                gp.GRB.EQUAL,
                wind_production[t, w] - production_DA[t],
                name=f"Delta definition with p_{t,w}^real and p_{t}^DA",
            )
            for w in range(nb_scenarios)
        }
        for t in range(24)
    }

    delta_value_pos_neg = {
        t: {
            w: m.addConstr(
                delta[t][w],
                gp.GRB.EQUAL,
                delta_up[t][w] - delta_down[t][w],
                name=f"Delta definition with delta_up and delta_down at time {t} for scenario {w}.",
            )
            for w in range(nb_scenarios)
        }
        for t in range(24)
    }

    delta_up_boundary = {
        t: {
            w: m.addConstr(
                delta_up[t][w],
                gp.GRB.LESS_EQUAL,
                P_nominal*binary[t][w],
                name=f"delta_up boundary at time {t} for scenario {w}.",
            )
            for w in range(nb_scenarios)
        }
        for t in range(24)
    }

    delta_down_boundary = {
        t: {
            w: m.addConstr(
                delta_down[t][w],
                gp.GRB.LESS_EQUAL,
                P_nominal*(1-binary[t][w]),
                name=f"delta_down boundary at time {t} for scenario {w}.",
            )
            for w in range(nb_scenarios)
        }
        for t in range(24)
    }
    
    # Optimize the model if specified
    if optimise:
        m.optimize()
        print(m.status == GRB.Status.OPTIMAL)
        prod = [production_DA[t].X for t in range(len(production_DA))]
        print(prod)
        plt.plot([i for i in range(len(production_DA))], prod)
        plt.show()

        # Export results if specified
        if m.status == 2 and export:
            export_results(m)
        elif m.status != 2 and export:
            print("Model have not converged - impossible to export results to json")
    else:
        m.update()

    return m, objective, production_DA, delta
