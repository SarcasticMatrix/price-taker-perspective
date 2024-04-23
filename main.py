from Step1.analysis import conduct_analysis 

from inputs.scenario_generator import scenarios_selection
in_sample_scenarios, out_sample_scenarios = scenarios_selection(seed=41)

########################################################################################################
###### Optimisation
########################################################################################################

### One Price Balancing Scheme
from Step1.onePriceBalancingScheme import onePriceBalancingScheme as OPBS
model = OPBS(scenarios=in_sample_scenarios, export=False)
expected_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='one')
print("Expected Profit:", expected_profit)

### Two Price Balancing Scheme
from Step1.twoPriceBalancingScheme import twoPriceBalancingScheme as TPBS
model = TPBS(scenarios=in_sample_scenarios, export=True)
expected_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios)
print("Expected Profit:", expected_profit)

########################################################################################################
###### Implementation of CVaR
########################################################################################################

### One Price Balancing Scheme
from Step1.CVaRModels import CVaR_onePriceBalancingScheme as CVaR_OPBS
model = CVaR_OPBS(alpha=0.05, beta=1, scenarios=in_sample_scenarios)
expected_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='two')
print("Expected Profit:", expected_profit)

## Two Price Balancing Scheme
from Step1.CVaRModels import CVaR_twoPriceBalancingScheme as CVaR_TPBS
model = CVaR_TPBS(alpha=0.05, beta=1, scenarios=in_sample_scenarios)
expected_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='two')
print("Expected Profit:", expected_profit)