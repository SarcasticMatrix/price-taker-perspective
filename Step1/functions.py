import pandas as pd 

def import_data():
    
    wind_scenarios = pd.read_csv(
        "inputs/data/scen_zoneW0.csv", delimiter=",", index_col=0
    )

    results = {
        "Wind scenarions": wind_scenarios
    }
    return results

if __name__ == "__main__":
    print(import_data())
