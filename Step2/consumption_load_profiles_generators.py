import numpy as np
import pandas as pd
import random

def Comsumption_profile_generator(seed:int = 42):
    """
    Generate a list of scenarios based on wind production, price, and power system need.

    Returns:
    - scenario_list (list): List containing DataFrames. Each element represent a scenario.
    """
    random.seed(seed)

    nbMin = 60
    nbScenarios = 201

    power_consumption = np.zeros((nbMin, nbScenarios))

    for i in range (nbScenarios):
        power_consumption[0,i] = np.random.uniform(200, 500)

    for i in range (1,nbMin):
        for j in range (nbScenarios):
            Delta=np.random.uniform(-25,25)
            Value=power_consumption[i-1,j]
            print(Value)
            while (Delta+Value>500) or (Delta+Value<200):
                Delta=np.random.uniform(-25,25)
            power_consumption[i,j]=Delta+power_consumption[i-1,j]
    df= pd.DataFrame(power_consumption)
    df.to_csv("consumption_load_profiles_scenarios.csv", index=False)
    return power_consumption


def scenarios_selection_Step2(load_scenarios_df, seed: int = 42, nbr_scenarios: int = 50) -> tuple:
    """
    Select 50 scenarios randomly from the generated scenarios list.

    Parameters:
    - seed (int): Seed of randomness for repeatability.

    Returns:
    - in_sample_scenarios (list): List containing 50 selected scenarios.
    - out_sample_scenarios (list): List containing the remaining scenarios after selection.
    """
    
    scenarios = []

    for load_scenario in load_scenarios_df.columns:
        combined_df = pd.DataFrame(
            {
                "Load profile": load_scenarios_df[load_scenario].values,
            }
        )
        scenarios.append(combined_df)
 
    in_sample_scenarios = scenarios[0:nbr_scenarios]
    out_sample_scenarios = scenarios[nbr_scenarios:]

    return in_sample_scenarios, out_sample_scenarios



if __name__ == "__main__":
    Comsumption_profile_generator()