import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random

from Step1.analysis import export_results

def onePriceBalancingScheme(
    scenarios: list, seed: int = 42, export: bool = False, optimise: bool = True
) -> gp.Model:
    """
    Implement model for the Offering Strategy Under a One-Price Balancing Scheme

    Inputs:
    - scenarios (list of pd.dataframe): list of all the 250 scenarios, obtained from the function 'scenarios_selection_250'
    - seed (int): seed for random number generation
    - export (bool): flag to export results to JSON
    - optimise (bool): flag to indicate whether to perform optimization

    Outputs:
    - m (gp.Model): optimized model
    """

    # Create a new model
    m = gp.Model("Offering Strategy Under a One-Price Balancing Scheme")

    ### Forecast inputs and model parameters
    P_nominal = 200
    pi = 1 / len(scenarios)
    price_DA = np.array(
        [scenarios[i]["Price DA"].values for i in range(len(scenarios))]
    )
    price_DA = np.transpose(price_DA)
    wind_production = P_nominal * np.array(
        [scenarios[i]["Wind production"].values for i in range(len(scenarios))]
    )
    wind_production = np.transpose(wind_production)
    power_needed = np.array(
        [scenarios[i]["Power system need"].values for i in range(len(scenarios))]
    )
    power_needed = np.transpose(power_needed)

    ### Variables
    # Define variables for power generation and forecast deviation
    production_DA = m.addMVar(
        shape=(24,), lb=0, ub=P_nominal, name="Power generation for 24 hours", vtype=GRB.CONTINUOUS
    )
    delta = m.addMVar(
        shape=(24, len(scenarios)), lb=-np.inf, name="Forecast deviation for 24 hours for 250 scenarios", vtype=GRB.CONTINUOUS
    )

    ### Objective function
    # Set the objective function
    objective = m.setObjective(
        sum(
            sum(
                pi
                * (
                    price_DA[t, w] * production_DA[t]
                    + (1 - power_needed[t, w]) * 0.9 * price_DA[t, w] * delta[t, w]
                    + power_needed[t, w] * 1.2 * price_DA[t, w] * delta[t, w]
                )
                for w in range(len(scenarios))
            )
            for t in range(24)
        ),
        GRB.MAXIMIZE,
    )

    ### Constraints
    # Define constraints on forecast deviation
    m.addConstrs(
        (
            delta[t, w] == wind_production[t, w] - production_DA[t]
            for t in range(24)
            for w in range(len(scenarios))
        ),
        name="Delta definition with p_{t,w}^real and p_t^DA",
    )

    # Optimize the model if specified
    if optimise:
        m.optimize()

        # Export results if specified
        if m.status == 2 and export:
            export_results(m)
        elif m.status != 2 and export:
            print("Model have not converged - impossible to export results to json")
    else:
        m.update()
        
    return m
