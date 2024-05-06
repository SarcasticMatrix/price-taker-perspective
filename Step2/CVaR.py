import gurobipy as gp
import numpy as np
from gurobipy import GRB
from gurobipy import quicksum

from Step2.analysis_bis import export_results

def CVaR(
    scenarios: list, export: bool = False, optimise: bool = True, epsilon: float = 0.1
) -> gp.Model:
    """
    Implement model for the Offering Strategy Under P90 requirement with CVaR method

    Inputs:
    - scenarios (list of pd.dataframe): list of all the 250 scenarios, obtained from the function 'scenarios_selection_250'
    - seed (int): seed for random number generation
    - export (bool): flag to export results to JSON
    - optimise (bool): flag to indicate whether to perform optimization
    - epsilon (float): 1- probability needed by the requirement

    Outputs:
    - model (gp.Model): optimized model
    """

    # Create a new model
    model = gp.Model("Offering Strategy Under P90 rule with the CVaR method")

    ### Forecast inputs and model parameters
    P_max_load = 500 #kW
    nbMin=60
    nbSamples=len(scenarios)
    F_up = np.array(
        [scenarios[i].values for i in range(nbSamples)]
    )

    ### Variables
    # Define variables for power generation and forecast deviation
    C_up = model.addVar(
        lb=0, ub=P_max_load, name="Optimal reserve capacity bid (in kW)", vtype=GRB.CONTINUOUS
    )
    beta = model.addVar(
        lb=-np.inf, ub=0, name="beta coefficient", vtype=GRB.CONTINUOUS
    )

    zeta = {
        m: {
            w: model.addVar(
                lb=-np.inf, name=f"zeta[{m,w}] (in kW)", vtype=GRB.CONTINUOUS
            )
            for w in range(nbSamples)
        }
        for m in range(nbMin)
    }

    ### Objective function
    # Set the objective function
    objective = model.setObjective(C_up, GRB.MAXIMIZE)

    ### Constraints
    # Define constraints on forecast deviation
    model.addConstr(
        (
            quicksum(
                quicksum(
                    zeta[m][w]
                for w in range(nbSamples)
                )
            for m in range(nbMin)
            )
        ) <= (1 - epsilon) * beta * nbMin * nbSamples,
        name="Probability definition",
    )


    for m in range(nbMin):
        for w in range(nbSamples):  
            model.addConstr(
                beta <= zeta[m][w],
                name=f"All the zeta above a threschold [{m,w}]",
            )

            model.addConstr(
                (
                    C_up - F_up[w,m] <= zeta[m][w]
                ),
                name=f"Delta between C_up and the production delivered below Zeta [{m,w}]",
            )


    # Optimize the model if specified
    if optimise:
        model.optimize()

        # Export results if specified
        if model.status == 2 and export:
            export_results(m)
        elif model.status != 2 and export:
            print("Model have not converged - impossible to export results to json")
    else:
        model.update()
        
    return model, C_up, beta, zeta