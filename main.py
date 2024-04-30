import numpy as np
import matplotlib.pyplot as plt

from Step1.analysis import compute_CVaR, compute_profits, conduct_analysis

from inputs.scenario_generator import scenarios_selection
in_sample_scenarios, out_sample_scenarios = scenarios_selection(seed=42)
import matplotlib.pyplot as plt

from Step1.onePriceBalancingScheme import onePriceBalancingScheme as OPBS
from Step1.twoPriceBalancingScheme import twoPriceBalancingScheme as TPBS
from Step1.CVaRModels import CVaR_onePriceBalancingScheme as CVaR_OPBS
from Step1.CVaRModels import CVaR_twoPriceBalancingScheme as CVaR_TPBS


########################################################################################################
###### Optimisation
########################################################################################################

### One Price Balancing Scheme
# model = OPBS(scenarios=in_sample_scenarios, export=False)
# expected_profit, standard_deviation_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='one')
# variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
# print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

# ### Two Price Balancing Scheme
# model, objective, production_DA, delta, delta_up, delta_down = TPBS(scenarios=in_sample_scenarios, export=True, optimise=True)
# expected_profit, standard_deviation_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='two', production_DA_dicc=production_DA, delta_dicc=delta, delta_up_dicc=delta_up, delta_down_dicc=delta_down)
# variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
# print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

########################################################################################################
###### Implementation of CVaR
########################################################################################################

### One Price Balancing Scheme
# model = CVaR_OPBS(alpha=0.95, beta=0.9, scenarios=in_sample_scenarios)
# expected_profit, standard_deviation_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme="one")
# variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
# print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

## Two Price Balancing Scheme
model, objective, production_DA, delta, delta_up, delta_down, eta, zeta = CVaR_TPBS(alpha=0.95, beta=0.5, scenarios=in_sample_scenarios)
expected_profit, standard_deviation_profit = conduct_analysis(m=model, scenarios=in_sample_scenarios, balancingScheme='two', production_DA_dicc=production_DA, delta_dicc=delta, delta_up_dicc=delta_up, delta_down_dicc=delta_down)
variation_coefficient  = round(standard_deviation_profit/expected_profit,2)
print(f"Expected profit: {round(expected_profit)}, Standard deviation: {round(standard_deviation_profit)}, variation_coefficient : {variation_coefficient }")

########################################################################################################
###### Plot of Expectde profit vs CVaR
########################################################################################################

alpha = 0.95
CVaRs = []
expected_profits = []
betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
balancingScheme = 'two'
for beta in betas:
    print(f'Working on beta:{beta}')
    if balancingScheme == 'one':
        model = CVaR_OPBS(alpha=alpha, beta=beta, scenarios=in_sample_scenarios)  #One Price Balancing Scheme
    else:
        model, objective, production_DA, delta, delta_up, delta_down, eta, zeta  = CVaR_TPBS(alpha=alpha, beta=beta, scenarios=in_sample_scenarios)  #Two Price Balancing Scheme
    CVaR = compute_CVaR(scenarios=in_sample_scenarios, model=model, alpha=alpha, balancingScheme=balancingScheme, production_DA_dicc=production_DA, delta_dicc=delta, delta_up_dicc=delta_up, delta_down_dicc=delta_down, eta_dicc=eta, zeta=zeta)
    profits = compute_profits(scenarios=in_sample_scenarios, m=model, balancingScheme=balancingScheme, production_DA_dicc=production_DA, delta_dicc=delta, delta_up_dicc=delta_up, delta_down_dicc=delta_down)
    expected_profit = np.mean(profits)
    expected_profits.append(expected_profit)
    CVaRs.append(CVaR)

expected_profits = np.array(expected_profits)
CVaRs = np.array(CVaRs)
print("CVARS", CVaRs, "expected_profits", expected_profits)

plt.figure()
plt.plot(CVaRs, expected_profits, label=rf'$\alpha={alpha}$')
plt.scatter(CVaRs, expected_profits, marker='*')
for i, txt in enumerate(betas):
#     plt.annotate(txt, betas[i])
    plt.annotate(txt, (CVaRs[i], expected_profits[i]))
plt.legend()
plt.grid(visible=True,which="major",linestyle="--", dashes=(5, 10), color="gray",linewidth=0.5,alpha=0.8)
plt.ylabel("Expected Profit [k€]")
plt.xlabel('CVaR [k€]')
plt.show()

########################################################################################################
###### Out-of-sample simulation
########################################################################################################

# model = OPBS(scenarios=in_sample_scenarios, export=False)
# profit_out_sample = compute_profits(scenarios=out_sample_scenarios, m=model, balancingScheme='one', nbr_scenarios=len(in_sample_scenarios))

# expected_profit_out_sample = np.mean(profit_out_sample)
# standard_deviation_out_profit_sample = np.std(profit_out_sample, ddof=1)

# profit_in_sample = compute_profits(scenarios=in_sample_scenarios, m=model, balancingScheme='one', nbr_scenarios=len(in_sample_scenarios))
# expected_profit_in_sample = np.mean(profit_in_sample)
# standard_deviation_in_sample = np.std(profit_in_sample, ddof=1)

# plt.figure()

# plt.hist(profit_in_sample/10**3, bins=20, edgecolor='None', color='red', alpha=0.3, label='In sample scenarios')
# plt.hist(profit_in_sample/10**3, bins=20, edgecolor="black", facecolor='None')
# plt.axvline(expected_profit_in_sample/10**3, color='purple', label='Expected profit with in sample scenarios')

# plt.hist(profit_out_sample/10**3, bins=20, edgecolor='None', color='orange', alpha=0.3, label='Out sample scenarios')
# plt.hist(profit_out_sample/10**3, bins=20, edgecolor="black", facecolor='None')
# plt.axvline(expected_profit_out_sample/10**3, color='blue', label='Expected profit with out sample scenarios')

# # plt.title(f"Profit Distribution Over Scenarios - Expected profit {round(expected_profit)} and its standard deviation {round(standard_deviation_profit)}")
# plt.xlabel("Profit (k€)")
# plt.minorticks_on()
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
# plt.grid(which='minor', visible=False)
# plt.show()
