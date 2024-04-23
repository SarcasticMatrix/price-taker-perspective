from inputs.scenario_generator import scenarios_selection_250
import Step1.onePriceBalancingScheme as opbs
import Step1.twoPriceBalancingScheme as tpbs

scenarios, _ = scenarios_selection_250()

### One Price Balancing Scheme
# model = opbs.onePriceBalancingScheme(scenarios=scenarios, export=False)
# expected_profit = opbs.conduct_analysis(m=model, scenarios=scenarios)
# print("Expected Profit:", expected_profit)

### Two Price Balancing Scheme
model = tpbs.twoPriceBalancingScheme(scenarios=scenarios, export=True)
expected_profit = tpbs.conduct_analysis(m=model, scenarios=scenarios)
print("Expected Profit:", expected_profit)