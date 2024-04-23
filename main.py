from inputs.scenario_generator import scenarios_selection
from Step1.onePriceBalancingScheme import onePriceBalancingScheme
from Step1.twoPriceBalancingScheme import twoPriceBalancingScheme
from Step1.analysis import conduct_analysis 

in_sample_scenarios, out_sample_scenarios = scenarios_selection()

########################################################################################################
###### Optimisation
########################################################################################################

### One Price Balancing Scheme
model = onePriceBalancingScheme(scenarios=in_sample_scenarios, export=False)
expected_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='one')
print("Expected Profit:", expected_profit)

### Two Price Balancing Scheme
model = twoPriceBalancingScheme(scenarios=in_sample_scenarios, export=True)
expected_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios)
print("Expected Profit:", expected_profit)

########################################################################################################
###### Implementation of CVaR
########################################################################################################

### One Price Balancing Scheme
model = onePriceBalancingScheme(scenarios=in_sample_scenarios, export=False)
expected_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='one')
print("Expected Profit:", expected_profit)

### Two Price Balancing Scheme
model = twoPriceBalancingScheme(scenarios=in_sample_scenarios, export=True)
expected_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios)
print("Expected Profit:", expected_profit)