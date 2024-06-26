import numpy as np
import pandas as pd
import random

wind_scenario_df = pd.read_csv("inputs/scen_zone1.csv", sep=";", index_col=0)
price_scenario_df = pd.read_csv("inputs/price_scenarios.csv", sep=";", index_col=0)
power_need_scenario_df = pd.read_csv("inputs/power_system_need_scenarios.csv", sep=",", index_col=0)


def imbalance_generator(
    seed: int = 42, deficit_probability: float = 0.5, export: bool = False
):
    """
    Generate a series of 24 random binary (two-state) variables,
    indicating in every hour of the next day, whether the system in the balancing stage will have a deficit in power supply or an excess.

    Parameters:
    - seed (int): Seed of randomness for repeatability.
    - deficit_probability (float): Probability of having a deficit in power supply in each hour.

    Returns:
    - power_needed (pd.DataFrame): DataFrame containing ones and zeros. 1 represents a deficit, 0 represents an excess.
    """

    random.seed(seed)

    nbHour = 24
    nbScenarios = 3

    power_needed = pd.DataFrame(index=range(0, nbHour))

    for i in range(1, nbScenarios + 1):
        state = (np.random.rand(nbHour) > deficit_probability).astype(int)
        power_needed[i] = state

    if export:
        print("Power need scenarios have been generated and are getting exported")
        csv_filename = "inputs/power_system_need_scenarios.csv"
        power_needed.to_csv(csv_filename, index=True)

    else:
        return power_needed


def scenarios_generator() -> list:
    """
    Generate a list of scenarios based on wind production, price, and power system need.

    Returns:
    - scenario_list (list): List containing DataFrames. Each element represent a scenario.
    """

    scenarios = []

    for wind_scenario in wind_scenario_df.columns:

        for price_scenario in price_scenario_df.columns:

            for power_need in power_need_scenario_df.columns:

                combined_df = pd.DataFrame(
                    {
                        "Wind production": wind_scenario_df[wind_scenario].values,
                        "Price DA": price_scenario_df[price_scenario].values,
                        "Power system need": power_need_scenario_df[power_need].values,
                    }
                )

                scenarios.append(combined_df)

    return scenarios


def scenarios_selection(seed: int = 42, nbr_scenarios: int = 250) -> tuple:
    """
    Select 250 scenarios randomly from the generated scenarios list.

    Parameters:
    - seed (int): Seed of randomness for repeatability.

    Returns:
    - in_sample_scenarios (list): List containing 250 selected scenarios.
    - out_sample_scenarios (list): List containing the remaining scenarios after selection.
    """

    scenarios = scenarios_generator()

    random.seed(seed)
    random.shuffle(scenarios)

    in_sample_scenarios = scenarios[0:nbr_scenarios]
    out_sample_scenarios = scenarios[nbr_scenarios:]

    return in_sample_scenarios, out_sample_scenarios


if __name__ == "__main__":

    imbalance_generator(export=True)
