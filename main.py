from Step1.analysis import conduct_analysis 

from inputs.scenario_generator import scenarios_selection
in_sample_scenarios, out_sample_scenarios = scenarios_selection(seed=42)

########################################################################################################
###### Optimisation
########################################################################################################

# ### One Price Balancing Scheme
# from Step1.onePriceBalancingScheme import onePriceBalancingScheme as OPBS
# model = OPBS(scenarios=in_sample_scenarios, export=False)
# expected_profit, standard_deviation_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='one')
# variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
# print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

# ### Two Price Balancing Scheme
# from Step1.twoPriceBalancingScheme import twoPriceBalancingScheme as TPBS
# model = TPBS(scenarios=in_sample_scenarios, export=True)
# expected_profit, standard_deviation_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='two')
# variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
# print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

# ########################################################################################################
# ###### Implementation of CVaR
# ########################################################################################################

### One Price Balancing Scheme
# from Step1.CVaRModels import CVaR_onePriceBalancingScheme as CVaR_OPBS
# model = CVaR_OPBS(alpha=0.95, beta=0.9, scenarios=in_sample_scenarios)
# expected_profit, standard_deviation_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme="one")
# variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
# print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

## Two Price Balancing Scheme
from Step1.CVaRModels import CVaR_twoPriceBalancingScheme as CVaR_TPBS
model = CVaR_TPBS(alpha=0.95, beta=0.5, scenarios=in_sample_scenarios)
expected_profit, standard_deviation_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='two')
variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

########################################################################################################
###### Plot of Expectde profit vs CVaR
########################################################################################################

import numpy as np
from Step1.CVaRModels import CVaR_onePriceBalancingScheme as CVaR_OPBS
from Step1.CVaRModels import CVaR_twoPriceBalancingScheme as CVaR_TPBS
from Step1.analysis import compute_CVaR, compute_profits

alpha = 0.95
CVaRs = []
expected_profits = []
betas = [0, 0.1, 0.5, 0.9, 1]
balancingScheme = 'two'
for beta in betas:
    print(f'Working on beta:{beta}')
    if balancingScheme == 'one':
        model = CVaR_OPBS(alpha=alpha, beta=beta, scenarios=in_sample_scenarios)  #One Price Balancing Scheme
    else:
        model = CVaR_TPBS(alpha=alpha, beta=beta, scenarios=in_sample_scenarios)  #Two Price Balancing Scheme
    CVaR = compute_CVaR(scenarios=in_sample_scenarios, model=model, alpha=alpha, balancingScheme=balancingScheme)
    profits = compute_profits(scenarios=in_sample_scenarios, m=model, balancingScheme=balancingScheme)
    expected_profit = np.mean(profits)
    expected_profits.append(expected_profit)
    CVaRs.append(CVaR)

expected_profits = np.array(expected_profits) / 10**3
CVaRs = np.array(CVaRs) / 10**3
print(CVaRs,expected_profits)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(CVaRs, expected_profits, label=rf'$\alpha={alpha}$')
plt.scatter(CVaRs, expected_profits, marker='*')
for i, txt in enumerate(betas):
    plt.annotate(txt, (CVaRs[i], expected_profits[i]))
plt.legend()
plt.grid(visible=True,which="major",linestyle="--", dashes=(5, 10), color="gray",linewidth=0.5,alpha=0.8)
plt.ylabel("Expected Profit [k€]")
plt.xlabel('CVaR [k€]')
plt.show()

