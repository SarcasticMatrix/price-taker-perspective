import gurobipy as gp
from gurobipy import GRB

def onePriceBalancingScheme(
        scnearios: list
) -> gp.model:
    
    # Create a new model
    m = gp.Model("Copper-plate single hour")

    # Create variables
    p = m.addMVar(
        shape=(12 + 6,), lb=0, ub=P_MAX, name="Power generation", vtype=GRB.CONTINUOUS
    )
    d = m.addMVar(
        shape=(3,), lb=0, ub=[0.5*D,0.4*D,0.1*D], name = "Demand", vtype=GRB.CONTINUOUS
    )

    objective = m.setObjective(sum(bid_price[i]*d[i] for i in range(3)) - sum(C[i] * p[i] for i in range(12 + 6)), GRB.MAXIMIZE)

    balance = m.addConstr(-sum(p[i] for i in range(12 + 6)) + sum(d[i] for i in range(3)) == 0, "Generation balance")

    m.optimize()

    return m