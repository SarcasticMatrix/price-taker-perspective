import pandas as pd

consumption_load_scenarios = pd.read_csv("Step2/consumption_load_profiles_scenarios.csv", sep=",")

from Step2.consumption_load_profiles_generators import scenarios_selection_Step2
in_sample_scenarios, out_sample_scenarios = scenarios_selection_Step2(consumption_load_scenarios)

########################################################################################################
###### Optimisation
########################################################################################################

### CVaR
from Step2.CVaR import CVaR
model, C_up_CVaR, beta, zeta = CVaR(scenarios=in_sample_scenarios)
C_up_CVaR = C_up_CVaR.x

### ALSO-X
from Step2.ALSOX import ALSOX
model, C_up_ALSOX, binary = ALSOX(scenarios=in_sample_scenarios)
C_up_ALSOX = C_up_ALSOX.x

from Step2.analysis import conduct_analysis
# conduct_analysis(in_sample_scenarios,C_up, binary)


########################################################################################################
###### Out of samples
########################################################################################################

from Step2.analysis import is_violated

violation_ratio = 0.1
testing_scenarios = in_sample_scenarios

results_CVaR = is_violated(testing_scenarios=testing_scenarios, C_up=C_up_CVaR)
violations_CVaR = 100 * sum(results_CVaR)/len(testing_scenarios)

results_ALSOX = is_violated(testing_scenarios=in_sample_scenarios, C_up=C_up_ALSOX)
violations_ALSOX = 100 * sum(results_ALSOX)/len(testing_scenarios)

number = int(100*(1-violation_ratio))
print(f"\nPercentage of P{number} violation is:\n     - CVaR: {violations_CVaR}%\n     - ALSO-X: {violations_ALSOX}%\n")