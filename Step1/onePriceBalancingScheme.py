import gurobipy as gp
import numpy as np

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
    - model_var_dic (dict): dictionary of variables from the model
    """

    # Create a new model
    m = gp.Model("Offering Strategy Under a One-Price Balancing Scheme")

    ### Forecast inputs and model parameters
    P_nominal = 200  # MW
    nb_scenarios = len(scenarios)
    pi = 1 / len(scenarios)
    price_DA = np.array(
        [scenarios[i]["Price DA"].values for i in range(nb_scenarios)]
    )
    price_DA = np.transpose(price_DA)
    wind_production = P_nominal * np.array(
        [scenarios[i]["Wind production"].values for i in range(nb_scenarios)]
    )
    wind_production = np.transpose(wind_production)
    power_needed = np.array(
        [scenarios[i]["Power system need"].values for i in range(nb_scenarios)]
    )
    power_needed = np.transpose(power_needed)

    ### Variables
    # Define variables for power generation and forecast deviation
    production_DA = {
        t: m.addVar(lb=0, ub=P_nominal, name=f"DA power generation at time {t}.")
        for t in range(24)
    }
    delta = {
        t: {
            w: m.addVar(
                lb=-gp.GRB.INFINITY,
                name=f"Forecast deviation at time {t} for scenario {w}.",
            )
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
                + (1 - power_needed[t, w]) * 0.9 * price_DA[t, w] * delta[t][w]
                + power_needed[t, w] * 1.2 * price_DA[t, w] * delta[t][w]
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
    
    model_var_dic = {
        "Objective": objective,
        "Production DA": production_DA,
        "Delta": delta
    }

    return m, model_var_dic
