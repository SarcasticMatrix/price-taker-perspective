import gurobipy as gp
import numpy as np
from gurobipy import GRB
from gurobipy import quicksum

def P90_CVaR(
    scenarios: list, seed: int = 42, export: bool = False, optimise: bool = True, epsilon: float = 0.1
) -> gp.Model:
    """
    Implement model for the Offering Strategy Under P90 requirement with CVaR method

    Inputs:
    - scenarios (list of pd.dataframe): list of all the 250 scenarios, obtained from the function 'scenarios_selection_250'
    - seed (int): seed for random number generation
    - export (bool): flag to export results to JSON
    - optimise (bool): flag to indicate whether to perform optimization
    - epsilon (float): (1- probability needed by the requirement)

    Outputs:
    - m (gp.Model): optimized model
    """

    # Create a new model
    m = gp.Model("Offering Strategy Under P90 rule with the CVaR method")

    ### Forecast inputs and model parameters
    P_max_load = 500 #kW
    nbMin=60
    nbSamples=len(scenarios)
    F_up = np.array(
        [scenarios[i].values for i in range(nbSamples)]
    )

    ### Variables
    # Define variables for power generation and forecast deviation
    C_up = m.addMVar(
        shape=(1,), lb=0, ub=P_max_load, name="Optimal reserve capacity bid (in kW)", vtype=GRB.CONTINUOUS
    )
    Beta = m.addMVar(
        shape=(1,), lb=-np.inf, ub=0, name="Beta coefficient", vtype=GRB.CONTINUOUS
    )
    Zeta = m.addMVar(
        shape=(nbMin,nbSamples), lb=-np.inf, name="Majorant of the delta between C_up and production Fm,w (in kW)", vtype=GRB.CONTINUOUS
    )

    ### Objective function
    # Set the objective function
    objective = m.setObjective(C_up,GRB.MAXIMIZE)

    ### Constraints
    # Define constraints on forecast deviation
    m.addConstr(
        (
            quicksum(
                quicksum(
                    Zeta[m,w]
                for w in range(nbSamples)
                )
            for m in range(nbMin)
            )
        )<=(1-epsilon)*Beta*(nbMin*nbSamples),
        name="Probability definition",
    )


    m.addConstrs(
        (
            Beta <= Zeta[m,w]
            for m in range(nbMin)
            for w in range(nbSamples)
        ),
        name="All the zeta above a threschold",
    )

    m.addConstrs(
        (
            C_up - F_up[w,m] <= Zeta[m,w]
            for m in range(nbMin)
            for w in range(nbSamples)
        ),
        name="Delta between C_up and the production delivered below Zeta",
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