import numpy as np
import pandas as pd
import random 

#%%
"""generate a series of 24 random binary (two-state) variables, e.g., using a bernoulli distribution, indicating in every hour of the next day, whether the system in the balancing stage will have a deficit in power supply or an excess."""
nbHour = 24
nbScenarios = 3
prob_deficit = 0.5  # 50% probability of deficit

hourly_state_df = pd.DataFrame(index=range(0,nbHour))
for i in range(1,nbScenarios+1):
    state = (np.random.rand(nbHour) > prob_deficit).astype(int)
    hourly_state_df[i] = state

csv_filename = 'power_system_need_scenarios.csv'
hourly_state_df.to_csv(csv_filename, index=True)


#%% Scenario generation

#extracting csv to dataframe
scen_zone1_df = pd.read_csv("scen_zone1.csv", sep=";", index_col=0)
price_scenario_df = pd.read_csv("price_scenarios.csv", sep=";", index_col=0)
power_system_need_scenarios_df = pd.read_csv("power_system_need_scenarios.csv", sep=",", index_col=0)

sceanario_list = []

def scenarios_generator():
    """
    Generate a list with the different scenarios 
    """
    # Iterate through all combinations of indices
    for wind_scenario in scen_zone1_df.columns:
        for price_scenario in price_scenario_df.columns:
            for power_need in power_system_need_scenarios_df.columns:
                # Create a new dataframe with one column for each scenario
                combined_df = pd.DataFrame({
                    'Wind production': scen_zone1_df[wind_scenario].values,
                    'Price': price_scenario_df[price_scenario].values,
                    'Power system need': power_system_need_scenarios_df[power_need].values
                })
                # Append the new dataframe to the list
                sceanario_list.append(combined_df)
    return(sceanario_list)

def scenario_selection_250():
    sceanario_list=scenarios_generator()
    random.seed(42)
    random.shuffle(sceanario_list)
    sceanario_list_250 = sceanario_list[0:250]
    return(sceanario_list_250)