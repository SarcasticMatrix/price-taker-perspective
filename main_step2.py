import pandas as pd

consumption_load_scenarios = pd.read_csv("Step2/consumption_load_profiles_scenarios.csv", sep=",", index_col=1)

from Step2.consumption_load_profiles_generators import scenarios_selection_Step2
in_sample_scenarios, out_sample_scenarios = scenarios_selection_Step2(consumption_load_scenarios)

########################################################################################################
###### Optimisation
########################################################################################################

### CVaR
from Step2.CVaR import CVaR
model, C_up, beta, zeta = CVaR(scenarios=in_sample_scenarios)

### ALSO-X
from Step2.ALSO_X import ALSOX
model, C_up, binary = ALSOX(scenarios=in_sample_scenarios)

#plots Step 2.1
from Step2.Analysis_step2 import conduct_analysis
conduct_analysis(in_sample_scenarios,C_up, binary)