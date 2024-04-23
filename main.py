from inputs.scenario_generator import scenarios_selection_250
from Step1.onePriceBalancingScheme import onePriceBalancingScheme, conduct_analysis
from Step1.twoPriceBalancingScheme import twoPriceBalancingScheme, conduct_analysis

scenarios, _ = scenarios_selection_250()

print(scenarios[0].head(10))

# One Price Balancing Scheme
#model = onePriceBalancingScheme(scenarios=scenarios, export=True)

# Two Price Balancing Scheme
model = twoPriceBalancingScheme(scenarios=scenarios)

print(model.status)
expected_profit = conduct_analysis(m=model, scenarios=scenarios)
print("Expected Profit:", expected_profit)