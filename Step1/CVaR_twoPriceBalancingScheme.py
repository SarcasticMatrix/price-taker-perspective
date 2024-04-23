import gurobipy as gp
from gurobipy import GRB
import numpy as np

from Step1.twoPriceBalancingScheme import twoPriceBalancingScheme

def CVaR_twoPriceBalancingScheme(
        scenarios: list, beta: float = 0.5, alpha: float = 0.5, seed: int = 42
) -> gp.Model:
    
    if not 0 <= beta <= 1 or not 0 <= alpha <= 1:
        raise ValueError("beta and alpha must be in [0,1].")

    model = twoPriceBalancingScheme(scenarios=scenarios, seed=seed, optimise=False)

    ### Forecasts inputs 

    price_DA = np.array(
        [scenarios[i]["Price DA"].values for i in range(len(scenarios))]
    )
    price_DA = np.transpose(price_DA)
    power_needed = np.array(
        [scenarios[i]["Power system need"].values for i in range(len(scenarios))]
    )
    power_needed = np.transpose(power_needed)

    ### Previous decision variables

    production_DA = [var for var in model.getVars() if "Power generation for 24 hours" in var.VarName]

    delta_up = [var for var in model.getVars() if "Upward forecast deviation for 24 hours for 250 scenarios" in var.VarName]
    delta_up = np.array(delta_up).reshape(24, len(scenarios))

    delta_down = [var for var in model.getVars() if "Downward forecast deviation for 24 hours for 250 scenarios" in var.VarName]
    delta_down = np.array(delta_down).reshape(24, len(scenarios))
    
    ### Add new decision Variables

    eta = model.addMVar(
        shape=(len(scenarios),), lb = 0, name="eta for 250 scenarios", vtype=GRB.CONTINUOUS,
    )

    zeta = model.addMVar(
        shape=(1,), lb = 0, name="zeta", vtype=GRB.CONTINUOUS,
    )

    ### New objective function

    obj_initial = model.getObjective()
    new_obj = (1 - beta) * obj_initial
    new_obj += beta * (
        zeta - 1 / (1-alpha) * sum(1/len(scenarios) * eta[w] for w in range(len(scenarios)) )
    )
    model.setObjective(new_obj, GRB.MAXIMIZE)

    ### Add new constraints
    model.addConstrs(
        (
        - sum(
                price_DA[t,w] * production_DA[t]
                + (1 - power_needed[t,w]) * price_DA[t,w] * (0.9 * delta_up[t,w] - delta_down[t,w])
                + power_needed[t,w] * price_DA[t,w] * (delta_up[t,w] - 1.2 * delta_down[t,w])
                for t in range(24)
            )
        + zeta - eta[w]  <= 0
        for w in range(len(scenarios))),
        name="equality constraints",
    )

    model.optimize()
    return model