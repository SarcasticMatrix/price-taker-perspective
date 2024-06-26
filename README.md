# Step 1:

Inputs of step 1 (inputs):
- power_system_need_scenarios.csv : 3 different scenarios of power needs (1 for deficit or 0 for excess)
- scen_zone1.csv : Imported wind availability data from https://www.renewables.ninja/ (20 days).
- price_scenarios.csv : Day-ahead market price from the 01/03/2024 to the 20/03/2024 from the Nordpool website (20 days).
- scenario_generator.py: Scenario generator to combine all the 1200 possible scenarios With the different price, wind availability and power system needs scenarios.

Code and function for step 1:
- onePriceBalancingScheme.py : one Price Balancing Scheme optimization model
- twoPriceBalancingScheme.py : two Price Balancing Scheme optimization model
- analysis.py : all the plots for step 1
- CVaRModels.py : Risk analysis

Code to compile: main_step1.py


# Step 2:

Inputs of step 2:
- consumption_load_load_profiles_scenarios.py: Scenario generator for the 200 load profiles scenarios following the requirements.
- consumption_load_load_profiles_scenarios.csv: Data of the load profile generated

Code and function for step 2:
- ALSO_X.py : ALSO-X method optimization model
- CVaR.py : CVaR method optimization model
- Validation.py : step 2.2
- analysis.py : all the plots of step 2.1 and 2.3

Code to compile: main_step1.py
