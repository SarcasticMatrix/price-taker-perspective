from inputs.scenario_generator import scenarios_selection_250
from Step1.onePriceBalancingScheme import onePriceBalancingScheme

scenarios, _ = scenarios_selection_250()

print(scenarios[0].head(10))

model = onePriceBalancingScheme(scenarios=scenarios)
print(model.status)

with open('model.txt', 'w') as f:
    # f.write("Variables:\n")
    # for var in model.getVars():
    #     f.write(f"{var.VarName}\n")
    
    f.write("Constraints:\n")
    for constr in model.getConstrs():
        f.write(f"{constr.ConstrName}\n")