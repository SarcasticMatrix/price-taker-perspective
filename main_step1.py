from Step1.analysis import conduct_analysis, plot_profit_all_cases, in_vs_out_plot_all_cases

from inputs.scenario_generator import scenarios_selection
in_sample_scenarios, out_sample_scenarios = scenarios_selection(seed=42)

from Step1.onePriceBalancingScheme import onePriceBalancingScheme as OPBS
from Step1.twoPriceBalancingScheme import twoPriceBalancingScheme as TPBS
from Step1.CVaRModels import CVaR_onePriceBalancingScheme as CVaR_OPBS
from Step1.CVaRModels import CVaR_twoPriceBalancingScheme as CVaR_TPBS


########################################################################################################
###### Optimisation
########################################################################################################

### One Price Balancing Scheme
model, model_var_dic = OPBS(scenarios=in_sample_scenarios)
expected_profit, standard_deviation_profit = conduct_analysis(m=model, model_var_dic=model_var_dic, scenarios=in_sample_scenarios, balancingScheme='one')
variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

### Two Price Balancing Scheme
model, model_var_dic = TPBS(scenarios=in_sample_scenarios)
expected_profit, standard_deviation_profit = conduct_analysis(m=model, model_var_dic=model_var_dic, scenarios=in_sample_scenarios, balancingScheme='two')
variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

#######################################################################################################
##### Implementation of CVaR
#######################################################################################################

### One Price Balancing Scheme
model, model_var_dic = CVaR_OPBS(alpha=0.95, beta=0.5, scenarios=in_sample_scenarios)
expected_profit, standard_deviation_profit = conduct_analysis(m=model, model_var_dic=model_var_dic, scenarios=in_sample_scenarios, balancingScheme="one")
variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

### Two Price Balancing Scheme
model, model_var_dic = CVaR_TPBS(alpha=0.95, beta=0.5, scenarios=in_sample_scenarios)
expected_profit, standard_deviation_profit = conduct_analysis(m=model, model_var_dic=model_var_dic, scenarios=in_sample_scenarios, balancingScheme='two')
variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

########################################################################################################
###### Normal vs CVaR profit distribution
########################################################################################################

plot_profit_all_cases(scenarios=in_sample_scenarios, beta=0.5, alpha=0.95)

########################################################################################################
###### Plot of Expectde profit vs CVaR
########################################################################################################

from Step1.analysis import expected_profit_vs_CVaR
alphas=[0.99]
balancingScheme = "one"
expected_profit_vs_CVaR(scenarios=in_sample_scenarios, alphas=alphas, balancingScheme=balancingScheme)

########################################################################################################
###### Out-of-sample simulation
########################################################################################################

in_vs_out_plot_all_cases(in_sample_scenarios=in_sample_scenarios, out_sample_scenarios=out_sample_scenarios)