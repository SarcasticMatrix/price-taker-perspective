import numpy as np
import pandas as pd

"""generate a series of 24 random binary (two-state) variables, e.g., using a bernoulli distribution, indicating in every hour of the next day, whether the system in the balancing stage will have a deficit in power supply or an excess."""

nbHour = 24
nbDays = 31
prob_deficit = 0.5  # 50% probability of deficit

hourly_state_df = pd.DataFrame(index=range(0,nbHour))
for i in range(1,nbDays):
    state = (np.random.rand(nbHour) > prob_deficit).astype(int)
    hourly_state_df[i] = state

csv_filename = 'Scenario_power_system_need.csv'
hourly_state_df.to_csv(csv_filename, index=True)

