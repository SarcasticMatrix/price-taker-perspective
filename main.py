from inputs.scenario_generator import scenarios_selection_250
from Step1.onePriceBalancingScheme import onePriceBalancingScheme

scenarios, _ = scenarios_selection_250()

print(scenarios[0].head())

model = onePriceBalancingScheme(scenarios=scenarios)
print(model.status)