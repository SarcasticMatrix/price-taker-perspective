import pandas as pd
consumption_load_scenario_df = pd.read_csv("Step2/consumption_load_profiles_scenarios.csv", sep=",", index_col=0)
print(consumption_load_scenario_df)
from Step2.consumption_load_profiles_generators import scenarios_selection_Step2
in_sample_scenarios, out_sample_scenarios = scenarios_selection_Step2(consumption_load_scenario_df)
from Step2.P90CVaR import P90_CVaR
model = P90_CVaR(scenarios=in_sample_scenarios)
from Step2.P90_ALSO_X import P90_ALSOX
model = P90_ALSOX(scenarios=in_sample_scenarios)