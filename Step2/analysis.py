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
    if len(shortfalls) > 0:
        shortfalls = np.hstack(shortfalls)
    else:
        shortfalls = [0]
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
    C_up_ALSOX: float, 
    binary: list,
    C_up_CVaR: float,
    Zeta: float 
):
    """
    Inputs:
    - scenarios (list of pd.dataframe): List of scenarios
    - C_up: optimal reserve capacity bid
    - binary: list of violations for the minutes and scenarios: 1 == violated
    """
    #print(zeta)
    # Retrieve optimaztion values
    nbSamples=len(scenarios)
    nbMin = 60
    Load_profile = np.array([scenarios[i].values for i in range(nbSamples)])

    Binary = np.zeros((nbSamples,nbMin))
    for i in range(nbSamples):
       for j in range(nbMin):
          Binary[i,j] = binary[j][i].x
    

    # Retrieve load profiles values
    load_random = Load_profile[2,:,0]
    load_random = np.hstack((load_random, load_random[-1]))
    Load_profile = np.sort(Load_profile, 0)
    load_max = Load_profile[-1, :,0]
    load_max = np.hstack((load_max, load_max[-1]))
    load_mean = Load_profile[:,:,0].mean(axis=0)
    load_mean = np.hstack((load_mean, load_mean[-1]))
    load_min = Load_profile[0, :,0]
    load_min = np.hstack((load_min, load_min[-1]))

    #Delta C_up_CVaR 
    C_up_CVaR_vect = C_up_CVaR*np.ones((nbMin+1))
    Delta_random = C_up_CVaR_vect - load_random
    Delta_max = C_up_CVaR_vect - load_max
    Delta_min = C_up_CVaR_vect - load_min
    Delta_mean = C_up_CVaR_vect - load_mean

    
    # Retrieve Violations values
    binary_random = Binary[2, :]
    binary_random = np.hstack((binary_random, binary_random[-1]))
    Binary = np.sort(Binary, 0)
    binary_max = Binary[-1, :]
    binary_max = np.hstack((binary_max, binary_max[-1]))
    binary_mean = Binary.mean(axis=0)
    binary_mean = np.hstack((binary_mean, binary_mean[-1]))
    binary_min = Binary[0, :]
    binary_min = np.hstack((binary_min, binary_min[-1]))

    # Retrieve Violations weighted values
    zeta_random = Zeta[2, :]
    zeta_random = np.hstack((zeta_random, zeta_random[-1]))
    Zeta = np.sort(Zeta, 0)
    zeta_max = Zeta[-1, :]
    zeta_max = np.hstack((zeta_max, zeta_max[-1]))
    zeta_mean = Zeta.mean(axis=0)
    zeta_mean = np.hstack((zeta_mean, zeta_mean[-1]))
    zeta_min = Zeta[0, :]
    zeta_min = np.hstack((zeta_min, zeta_min[-1]))
    
    
    # Create time array
    min = [i for i in range(61)]

    #################################################################################
    ### ALSO-X ###
    # Plotting
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.5]}
    )

    # Plot consumption load profile and optimal reserve capacity bid
    ax1.step(
        min, load_min, color="purple", linestyle="dotted", where="post", linewidth=1
    )
    ax1.step(
        min, load_mean, color="purple", linestyle="dashed", where="post", linewidth=1
    )
    ax1.step(
        min, load_max, color="purple", linestyle="dotted", where="post", linewidth=1
    )
    ax1.step(
        min, load_random, color="purple", linestyle="solid", where="post", label=r"$F_{m,w_0}$", linewidth=1
    )
    ax1.step(min, [C_up_ALSOX for i in range(61)], label=r"$C_{ALSO-X}$", where="post", color="blue", linewidth=0.7)
    ax1.step(min, [C_up_CVaR for i in range(61)], label=r"$C_{CVaR}$", where="post", color="red", linewidth=0.7)
    ax1.set_title(r"Consumption load profile $F_{m,w}$ and optimal reserve capacity bid $C_{up}$")
    ax1.set_ylabel("Load profile [kW]")
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

    
    # Plot system violation 
    ax2.step(
        min,
        binary_min,
        color="green",
        linestyle="dotted",
        where="post",
        linewidth=1,
    )
    ax2.step(
        min,
        binary_max,
        color="green",
        linestyle="dotted",
        where="post",
        linewidth=1,
    )
    ax2.step(
        min,
        binary_mean,
        color="green",
        linestyle="dashed",
        where="post",
        linewidth=1,
    )
    ax2.step(
        min,
        binary_random,
        color="green",
        linestyle="solid",
        where="post",
        label=r"$1 - y_{m,w_0}$",
        linewidth=1,
    )
    ax2.set_title(r"System violations (ALSO-X method)")
    ax2.set_xlabel("Minutes")
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
    label_axis_x = ["" for i in range(61)]
    for i in range(13):
        label_axis_x [5*i]= f"{5*i}"
    plt.xticks(min, label_axis_x)
    plt.show()

    #################################################################################
    ### CVaR ###
    # Plotting
    fig, (axi1, axi2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.5]}
    )

    # Plot production DA and wind production forecast
    axi1.step(
        min, load_min, color="purple", linestyle="dotted", where="post", linewidth=1
    )
    axi1.step(
        min, load_mean, color="purple", linestyle="dashed", where="post", linewidth=1
    )
    axi1.step(
        min, load_max, color="purple", linestyle="dotted", where="post", linewidth=1
    )
    axi1.step(
        min, load_random, color="purple", linestyle="solid", where="post", label=r"$F_{m,w_0}$", linewidth=1
    )
    axi1.step(min, [C_up_CVaR for i in range(61)], label=r"$C_{up,CVaR}$", where="post", color="red", linewidth=0.7)
    axi1.set_title(r"Consumption load profile $F_{m,w}$ and optimal reserve capacity bid $C_{up}$")
    axi1.set_ylabel("Load profile [kW]")
    axi1.grid(
        visible=True,
        which="major",
        linestyle="--",
        dashes=(5, 10),
        color="gray",
        linewidth=0.5,
        alpha=0.8,
    )
    axi1.grid(which="minor", visible=False)
    axi1.legend(loc="upper left")

    # Plot system violation weighted
    axi2.step(min, [0 for i in range(61)],  where="post", color="black", linewidth=0.7)
    axi2.step(
        min,
        Delta_min,
        color="green",
        linestyle="dotted",
        where="post",
        linewidth=1,
    )
    axi2.step(
        min,
        zeta_min,
        color="purple",
        linestyle="dotted",
        where="post",
        linewidth=1,
    )
    axi2.step(
        min,
        zeta_max,
        color="purple",
        linestyle="dotted",
        where="post",
        linewidth=1,
    )
    axi2.step(
        min,
        zeta_mean,
        color="purple",
        linestyle="dashed",
        where="post",
        linewidth=1,
    )
    axi2.step(
        min,
        Delta_random,
        color="green",
        linestyle="solid",
        where="post",
        label=r"$\zeta _{m,w_0}$",
        linewidth=1,
    )
    axi2.step(
        min,
        zeta_random,
        color="purple",
        linestyle="solid",
        label=r"$C_{up} - F_{m,w_0}$",
        where="post",
        linewidth=1,
    )
    axi2.set_title("System violations power weighted (CVaR method)")
    axi2.grid(
        visible=True,
        which="major",
        linestyle="--",
        dashes=(5, 10),
        color="gray",
        linewidth=0.5,
        alpha=0.8,
    )
    axi2.grid(which="minor", visible=False)
    axi2.legend(loc="lower left")
    label_axis_x = ["" for i in range(61)]
    for i in range(13):
        label_axis_x[5*i] = f"{5*i}"
    plt.xticks(min, label_axis_x)
    plt.show()


import matplotlib.pyplot as plt
from tqdm import tqdm
from Step2.ALSOX import ALSOX
from Step2.CVaR import CVaR
import sys
import os
def c_vs_ES(
        in_sample_scenarios: list,
        testing_scenarios: list,
): 
    epsilons = [0, 0.025, 0.05, 0.1, 0.15, 0.2]
    C_up_CVaRs = []
    expected_shortfalls_CVaRs = []
    violations_CVaRs = []

    C_up_ALSOXs = []
    expected_shortfalls_ALSOXs = []
    violations_ALSOXs = []

    for epsilon in tqdm(epsilons):

        # CVaR
        sys.stdout = open(os.devnull, 'w')
        model, C_up_CVaR, beta, zeta = CVaR(scenarios=in_sample_scenarios, epsilon=epsilon)
        C_up_CVaR = C_up_CVaR.x
        sys.stdout = sys.__stdout__

        violation, expected_shortfall = is_violated(testing_scenarios=testing_scenarios, C_up=C_up_CVaR)
        C_up_CVaRs.append(C_up_CVaR)
        expected_shortfalls_CVaRs.append(expected_shortfall)
        violations_CVaRs.append(violation)

        # ALSO-X
        sys.stdout = open(os.devnull, 'w')
        model, C_up_ALSOX, binary = ALSOX(scenarios=in_sample_scenarios, epsilon=epsilon)
        C_up_ALSOX = C_up_ALSOX.x
        sys.stdout = sys.__stdout__

        violation, expected_shortfall = is_violated(testing_scenarios=testing_scenarios, C_up=C_up_ALSOX)
        C_up_ALSOXs.append(C_up_ALSOX)
        expected_shortfalls_ALSOXs.append(expected_shortfall)
        violations_ALSOXs.append(violation)

    import numpy as np
    numbers = [int(100*(1-epsilon)) for epsilon in epsilons]
    numbers = np.array(numbers)

    C_up_CVaR = np.array(C_up_CVaRs)
    expected_shortfalls_CVaRs = np.array(expected_shortfalls_CVaRs)

    C_up_ALSOX = np.array(C_up_ALSOXs)
    expected_shortfalls_ALSOXs = np.array(expected_shortfalls_ALSOXs)

    fig = plt.figure()

    # Subplot 1: CVaR and ALSOX as a function of numbers
    ax1a = plt.subplot2grid((4, 2), (0, 0), rowspan=3)
    ax1a.plot(numbers, C_up_CVaRs, label='CVaR', color='orange')
    ax1a.scatter(numbers, C_up_CVaRs, color='orange')
    ax1a.plot(numbers, C_up_ALSOXs, label='ALSOX', color='blue')
    ax1a.scatter(numbers, C_up_ALSOXs, color='blue')

    ax1a.plot(numbers, expected_shortfalls_CVaRs, color='orange', linestyle='--')
    ax1a.scatter(numbers, expected_shortfalls_CVaRs, color='orange')
    ax1a.plot(numbers, expected_shortfalls_ALSOXs, color='blue', linestyle='--')
    ax1a.scatter(numbers, expected_shortfalls_ALSOXs, color='blue')

    ax1a.legend()
    ax1a.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
    ax1a.set_title("Optimal reserve bid over rule")
    ax1a.set_ylabel("Reserve bid [kW]")
    ax1a.set_xticks(numbers)
    ax1a.set_xticklabels([f"P{number}" for number in numbers])

    ax1b = plt.subplot2grid((4, 2), (3, 0), rowspan=1)
    ax1b.plot(numbers, violations_CVaRs, color='orange', linewidth=0.9)
    ax1b.plot(numbers, violations_ALSOXs, color='blue', linewidth=0.9)
    ax1b.set_ylabel("Number of violations [%]")
    ax1b.set_xlabel("Rule")
    ax1b.set_yticks([0, 5, 10, 15])
    ax1a.set_title("Number of violaiton over rule")
    ax1b.set_xticks(numbers)
    ax1b.set_xticklabels([f"P{number}" for number in numbers])
    ax1b.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)

    # Subplot 2: Expected shortfall as a function of C_up
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax2.plot(C_up_CVaRs, expected_shortfalls_CVaRs, label='CVaR', color='orange')
    ax2.scatter(C_up_CVaRs, expected_shortfalls_CVaRs, color='orange')
    ax2.plot(C_up_ALSOXs, expected_shortfalls_ALSOXs, label='ALSOX', color='blue')
    ax2.scatter(C_up_ALSOXs, expected_shortfalls_ALSOXs, color='blue')
    ax2.legend()
    ax2.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
    ax2.set_title("Expected reserve shortfall over reserve bid")
    ax2.set_ylabel("Expected shortfall [kW]")
    ax2.set_xlabel("Reserve bid [kW]")

    plt.tight_layout()
    plt.show()
