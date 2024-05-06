import numpy as np
import matplotlib.pyplot as plt
import json
import gurobipy as gp

def export_results(model: gp.Model):
    """
    Export the results to 'result.json'.

    Inputs:
    - model (gp.Model): Optimized model
    """
    # Retrieve variable values and names
    values = model.getAttr("X", model.getVars())
    names = model.getAttr("VarName", model.getVars())
    name_to_value = {name: value for name, value in zip(names, values)}

    # Export results to JSON
    with open("result.json", "w") as f:
        json.dump(name_to_value, f, indent=1)

def is_violated(
        testing_scenarios:list, 
        C_up:float, 
    ):
    """
    Utilise les scenarios pour back test un C_up optimisé
    """

    nbr_samples = len(testing_scenarios)
    results = []
    shortfalls = []

    # somme sur les scénarios w
    for profile in testing_scenarios:

        mask = profile['Load profile'] < C_up
        count = np.sum(mask.values)
        results.append(count)

        deviation = C_up - profile['Load profile']
        shortfall = deviation.loc[mask,].values
        if len(shortfall) > 0:
            shortfalls.append(shortfall)    
    
    # somme les minutes m
    results = np.sum(results)
    violations = 100 * results / (60 * nbr_samples)

    # Expected shortfalls
    shortfalls = np.hstack(shortfalls)
    expected_shortfall = np.mean(shortfalls)
    
    return violations, expected_shortfall

import matplotlib.pyplot as plt

def cross_validation(
        testing_scenarios: list,
        C_up_CVaR: float,
        C_up_ALSOX: float,
        violation_ratio: float = 0.1
):
    nbr_scenarios = len(testing_scenarios)

    results_CVaR = is_violated(testing_scenarios=testing_scenarios, C_up=C_up_CVaR, violation_ratio=violation_ratio)
    violations_CVaR = 100 * sum(results_CVaR)/nbr_scenarios

    results_ALSOX = is_violated(testing_scenarios=testing_scenarios, C_up=C_up_ALSOX, violation_ratio=violation_ratio)
    violations_ALSOX = 100 * sum(results_ALSOX)/nbr_scenarios

    methods = ['CVaR', 'ALSO-X']
    violations = [violations_CVaR, violations_ALSOX]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    axs[0].bar(methods[0], violations[0], width=0.1)
    axs[0].set_ylabel('% of violations', fontweight='bold')
    axs[0].grid(axis='y', linestyle='--', dashes=(5, 10), color='gray', linewidth=0.5, alpha=0.8)
    axs[0].tick_params(axis='x')
    axs[0].set_title(f'CVaR: {violations[0]:.2f}% violations')

    axs[1].bar(methods[1], violations[1], width=0.1)
    axs[1].grid(axis='y', linestyle='--', dashes=(5, 10), color='gray', linewidth=0.5, alpha=0.8)
    axs[1].tick_params(axis='x')
    axs[1].set_title(f'ALSO-X: {violations[1]:.2f}% violations')

    number = int(100*(1-violation_ratio))
    fig.suptitle(f'Number of P{number} violations with {nbr_scenarios} testing profiles', fontweight='bold')

    plt.tight_layout()
    plt.show()



def conduct_analysis(
    scenarios: list,
    C_up: float, 
    binary: list
):
    """
    Inputs:
    - scenarios (list of pd.dataframe): List of scenarios
    - C_up: optimal reserve capacity bid
    - binary: list of violations for the minutes and scenarios: 1 == violated
    """
    # Retrieve price scenarios values
    nbSamples=len(scenarios)
    nbMin = 60
    Load_profile = np.array([scenarios[i].values for i in range(nbSamples)])
    #Load_profile = np.zeros((nbSamples,nbMin))
    #for i in range (nbSamples):
        #Load_profile[i]=scenarios[i].values
    Binary = np.zeros((nbSamples,nbMin))
    for i in range(nbSamples):
       for j in range(nbMin):
          Binary[i,j] = binary[j][i].x

    # Retrieve load profiles values
    load_random = Load_profile[20,:,0]
    load_random = np.hstack((load_random, load_random[-1]))
    Binary = np.sort(Binary, 0)
    load_max = Load_profile[-1, :,0]
    load_max = np.hstack((load_max, load_max[-1]))
    load_mean = Load_profile[:,:,0].mean(axis=0)
    load_mean = np.hstack((load_mean, load_mean[-1]))
    load_min = Load_profile[0, :,0]
    load_min = np.hstack((load_min, load_min[-1]))
    
    # Retrieve Violations values
    binary_random = Binary[20, :]
    binary_random = np.hstack((binary_random, binary_random[-1]))
    Binary = np.sort(Binary, 0)
    binary_max = Binary[-1, :]
    binary_max = np.hstack((binary_max, binary_max[-1]))
    binary_mean = Binary.mean(axis=0)
    binary_mean = np.hstack((binary_mean, binary_mean[-1]))
    binary_min = Binary[0, :]
    binary_min = np.hstack((binary_min, binary_min[-1]))
    
    
    # Create time array
    min = [i for i in range(61)]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.5]}
    )

    # Plot production DA and wind production forecast
    ax1.step(
        min, load_min, color="purple", linestyle="dotted", where="post", linewidth=1
    )
    ax1.step(
        min, load_mean, color="purple", linestyle="solid", where="post", linewidth=1
    )
    ax1.step(
        min, load_max, color="purple", linestyle="dashed", where="post", linewidth=1
    )
    ax1.step(
        min, load_random, color="purple", linestyle="dashed", where="post", linewidth=1
    )
    #ax1.step(
        #np.nan,
        #np.nan,
        #color="purple",
        #linestyle="solid",
        #label="Power availability",
        #where="post",
        #linewidth=0.7,
    #)
    #ax1.step(, production_DA, label=r"$p_{t}^{DA}$", where="post", color="red")
    ax1.set_title(
        r"DA offered production $p_t^{DA}$ and wind power forecast $p_{t,w}^{real}$"
    )
    ax1.set_ylabel("Power [kW]")
    ax1.grid(
        visible=True,
        which="major",
        linestyle="--",
        dashes=(5, 10),
        color="gray",
        linewidth=0.5,
        alpha=0.8,
    )
    ax1.grid(which="minor", visible=False)
    ax1.legend(loc="upper left")

    
    # Plot power system need
    ax2.step(
        min,
        binary_min,
        color="green",
        linestyle="solid",
        where="post",
        label=r"$x_{t,w}^B$",
        linewidth=1,
    )
    ax2.step(
        min,
        binary_max,
        color="green",
        linestyle="solid",
        where="post",
        linewidth=1,
    )
    ax2.step(
        min,
        binary_mean,
        color="green",
        linestyle="dotted",
        where="post",
        linewidth=1,
    )
    ax2.step(
        min,
        binary_random,
        color="green",
        linestyle="dotted",
        where="post",
        linewidth=1,
    )
    #ax2.step(
        #np.nan,
        #np.nan,
        #color="green",
        #linestyle="solid",
        #label=r"$x_{t,w}^B$",
        #linewidth=0.7,
    #)
    ax2.set_title(r"Power system need $x_{t,w}^B$")
    ax2.set_xlabel("Minutes")
    ax2.set_yticks([0, 1], ["Non-Violated", "Violated"])
    ax2.grid(
        visible=True,
        which="major",
        linestyle="--",
        dashes=(5, 10),
        color="gray",
        linewidth=0.5,
        alpha=0.8,
    )
    ax2.grid(which="minor", visible=False)
    ax2.legend(loc="upper left")

    plt.xticks(min, [f"m{i}" for i in range(60)] + ["H0"])
    plt.show()

    return 