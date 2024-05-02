import matplotlib.pyplot as plt
import gurobipy as gp
import numpy as np
from typing import Literal

import json


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


def compute_CVaR(
    scenarios: list,
    model: gp.Model,
    alpha: float = 0.95,
    balancingScheme: Literal["one", "two"] = "two",
    production_DA_dicc=None,
    delta_dicc=None,
    delta_up_dicc=None,
    delta_down_dicc=None,
    eta_dicc=None,
    zeta=None,
) -> float:
    """
    Compute the Conditional Value-at-Risk (CVaR), which represents the expected profit value of the worst scenarios
    constituting $(1-\alpha) \times 100 \%$ of the profit distribution.

    Parameters:
        scenarios (list): A list of scenarios.
        model (gp.Model): The optimized Gurobi model.
        alpha (float): The confidence level, indicating the 100 - percentage of worst scenarios considered.
        balancingScheme (Literal["one", "two"]): Type of balancing scheme ('one' or 'two')

    Returns:
        float: The computed CVaR.
    """

    profits = compute_profits(
        scenarios,
        model,
        balancingScheme,
        len(scenarios),
        production_DA_dicc,
        delta_dicc,
        delta_up_dicc,
        delta_down_dicc,
    )
    sorted_profits = sorted(profits)
    alpha_index = int(len(sorted_profits) * (1 - alpha)) + 1
    smallest_profits = sorted_profits[:alpha_index]
    CVaR = np.mean(smallest_profits)

    # eta = [eta_dicc[w].x for w in range(len(scenarios))]
    # eta = np.array(eta)
    # zeta = zeta.x
    # CVaR = zeta - 1 / (1 - alpha) * np.sum(eta) * 1 / len(scenarios)

    return CVaR


def compute_profits(
    scenarios: list,
    m: gp.Model,
    balancingScheme: Literal["one", "two"] = "two",
    nbr_scenarios=250,
    production_DA_dicc=None,
    delta_dicc=None,
    delta_up_dicc=None,
    delta_down_dicc=None,
):
    """
    Compute the profit based on the optimized model and scenarios.

    Parameters:
        scenarios (list): A list of scenarios.
        m (gp.Model): The optimized Gurobi model.
        balancingScheme (Literal["one", "two"]): Type of balancing scheme ('one' or 'two')


    Returns:
        np.array: The profits for each scenarios.
    """

    # Retrieve production DA values
    if balancingScheme == "one":
        production_DA = [
            var.X
            for var in m.getVars()
            if "Power generation for 24 hours" in var.VarName
        ]
    else:
        production_DA = [production_DA_dicc[t].x for t in range(24)]

    # Retrieve price DA values
    price_DA = np.array([scenarios[i]["Price DA"].values for i in range(nbr_scenarios)])
    price_DA = np.transpose(price_DA)

    power_needed = np.array(
        [scenarios[i]["Power system need"].values for i in range(nbr_scenarios)]
    )
    power_needed = np.transpose(power_needed)

    if balancingScheme == "one":
        delta = [
            var.X
            for var in m.getVars()
            if "Forecast deviation for 24 hours for 250 scenarios" in var.VarName
        ]
        delta = np.array(delta).reshape(24, nbr_scenarios)

    else:
        delta_up = [
            [delta_up_dicc[t][w].x for w in range(nbr_scenarios)] for t in range(24)
        ]
        delta_down = [
            [delta_down_dicc[t][w].x for w in range(nbr_scenarios)] for t in range(24)
        ]

    profits = []
    if balancingScheme == "one":
        for w in range(nbr_scenarios):
            profit_w = sum(
                (
                    price_DA[t, w] * production_DA[t]
                    + (1 - power_needed[t, w]) * 0.9 * price_DA[t, w] * delta[t, w]
                    + power_needed[t, w] * 1.2 * price_DA[t, w] * delta[t, w]
                )
                for t in range(24)
            )
            profits.append(profit_w)

    else:
        for w in range(nbr_scenarios):
            profit_w = sum(
                (
                    price_DA[t, w] * production_DA[t]
                    + (1 - power_needed[t, w]) * price_DA[t, w] * (0.9 * delta_up[t][w] - delta_down[t][w])
                    + power_needed[t, w] * price_DA[t, w] * (delta_up[t][w] - 1.2 * delta_down[t][w])
                )
                for t in range(24)
            )
            profits.append(profit_w)

    profits = np.array(profits)
    return profits


def conduct_analysis(
    scenarios: list,
    m: gp.Model,
    balancingScheme: Literal["one", "two"] = "two",
    production_DA_dicc=None,
    delta_dicc=None,
    delta_up_dicc=None,
    delta_down_dicc=None,
):
    """
    Analyzes the results of the optimization model for the offering strategy under a one-price balancing scheme.

    Inputs:
    - scenarios (list of pd.dataframe): List of scenarios
    - m (gp.Model): Optimized model
    - balancingScheme (Literal["one", "two"]): Type of balancing scheme ('one' or 'two')

    Returns:
    - expected_profit (float): Expected profit
    - standard_deviation_profit (float) Standard deviation of profit
    """
    # Retrieve production DA values
    if balancingScheme == "one":
        production_DA = [
            var.X
            for var in m.getVars()
            if "Power generation for 24 hours" in var.VarName
        ]
    else:
        production_DA = [production_DA_dicc[t].x for t in range(24)]

    production_DA = np.hstack((production_DA, production_DA[-1]))
    # Retrieve price DA values
    price_DA = np.array(
        [scenarios[i]["Price DA"].values for i in range(len(scenarios))]
    )
    price_DA = np.transpose(price_DA)

    # Retrieve delta values
    if balancingScheme == "one":
        delta = [
            var.X
            for var in m.getVars()
            if "Forecast deviation for 24 hours for 250 scenarios" in var.VarName
        ]
        delta = np.array(delta).reshape(24, len(scenarios)).T
    else:
        delta = [[delta_dicc[t][w].x for w in range(len(scenarios))] for t in range(24)]
        delta = np.array(delta).T
    delta = np.sort(delta, 0)
    delta_max = delta[-1, :]
    delta_max = np.hstack((delta_max, delta_max[-1]))
    delta_mean = delta.mean(axis=0)
    delta_mean = np.hstack((delta_mean, delta_mean[-1]))
    delta_min = delta[0, :]
    delta_min = np.hstack((delta_min, delta_min[-1]))

    # Retrieve wind production forecast values
    P_nominal = 200
    wind_production_forecast = P_nominal * np.array(
        [scenarios[i]["Wind production"].values for i in range(len(scenarios))]
    )
    wind_production_forecast = np.sort(wind_production_forecast, 0)
    wind_max = wind_production_forecast[-1, :]
    wind_max = np.hstack((wind_max, wind_max[-1]))
    wind_mean = wind_production_forecast.mean(axis=0)
    wind_mean = np.hstack((wind_mean, wind_mean[-1]))
    wind_min = wind_production_forecast[0, :]
    wind_min = np.hstack((wind_min, wind_min[-1]))

    # Retrieve power system need values
    power_system_need = np.array(
        [scenarios[i]["Power system need"].values for i in range(len(scenarios))]
    )
    power_system_need = np.sort(power_system_need, 0)
    power_system_need_max = power_system_need[-1, :]
    power_system_need_max = np.hstack(
        (power_system_need_max, power_system_need_max[-1])
    )
    power_system_need_mean = power_system_need.mean(axis=0)
    power_system_need_mean = np.hstack(
        (power_system_need_mean, power_system_need_mean[-1])
    )
    power_system_need_min = power_system_need[0, :]
    power_system_need_min = np.hstack(
        (power_system_need_min, power_system_need_min[-1])
    )

    # Create time array
    time = [i for i in range(25)]

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5]}
    )

    # Plot production DA and wind production forecast
    ax1.step(
        time, wind_max, color="purple", linestyle="dotted", where="post", linewidth=1
    )
    ax1.step(
        time, wind_mean, color="purple", linestyle="solid", where="post", linewidth=1
    )
    ax1.step(
        time, wind_min, color="purple", linestyle="dashed", where="post", linewidth=1
    )
    ax1.step(
        np.nan,
        np.nan,
        color="purple",
        linestyle="solid",
        label="Power availability",
        where="post",
        linewidth=0.7,
    )
    ax1.step(time, production_DA, label=r"$p_{t}^{DA}$", where="post", color="red")
    ax1.set_title(
        r"DA offered production $p_t^{DA}$ and wind power forecast $p_{t,w}^{real}$"
    )
    ax1.set_ylabel("Power [MW]")
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

    # Plot delta
    ax2.step(
        time, delta_max, color="blue", linestyle="dotted", where="post", linewidth=1
    )
    ax2.step(
        time, delta_mean, color="blue", linestyle="solid", where="post", linewidth=1
    )
    ax2.step(
        time, delta_min, color="blue", linestyle="dashed", where="post", linewidth=1
    )
    ax2.step(
        np.nan,
        np.nan,
        color="blue",
        linestyle="solid",
        label=r"$\Delta_{t,w}$",
        linewidth=0.7,
    )
    ax2.set_title(r"Planned power deviation from forecasts $\Delta_{t,w}$")
    ax2.set_ylabel("Power [MW]")
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

    # Plot power system need
    ax3.step(
        time,
        power_system_need_max,
        color="green",
        linestyle="dotted",
        where="post",
        linewidth=1,
    )
    ax3.step(
        time,
        power_system_need_mean,
        color="green",
        linestyle="solid",
        where="post",
        linewidth=1,
    )
    ax3.step(
        time,
        power_system_need_min,
        color="green",
        linestyle="dashed",
        where="post",
        linewidth=1,
    )
    ax3.step(
        np.nan,
        np.nan,
        color="green",
        linestyle="solid",
        label=r"$x_{t,w}^B$",
        linewidth=0.7,
    )
    if balancingScheme == "one":
        ax3.axhline(1 / 3, label="threshold", color="red", linestyle="dashed")
    ax3.set_title(r"Power system need $x_{t,w}^B$")
    ax3.set_xlabel("Hours")
    ax3.set_yticks([0, 1], ["Excess", "Deficit"])
    ax3.grid(
        visible=True,
        which="major",
        linestyle="--",
        dashes=(5, 10),
        color="gray",
        linewidth=0.5,
        alpha=0.8,
    )
    ax3.grid(which="minor", visible=False)
    ax3.legend(loc="upper left")

    plt.xticks(time, [f"H{i}" for i in range(24)] + ["H0"])
    plt.show()

    # Plot profit distribution
    profits = compute_profits(
        scenarios,
        m,
        balancingScheme,
        production_DA_dicc=production_DA_dicc,
        delta_dicc=delta_dicc,
        delta_up_dicc=delta_up_dicc,
        delta_down_dicc=delta_down_dicc,
    )
    expected_profit = np.mean(profits)
    standard_deviation_profit = np.std(profits, ddof=1)

    plt.figure()
    plt.hist(profits / 10**3, bins=20, edgecolor="None", color="red", alpha=0.3)
    plt.hist(profits / 10**3, bins=20, edgecolor="black", facecolor="None")
    plt.axvline(expected_profit / 10**3, color="purple", label="Expected profit")
    plt.title(
        f"Profit Distribution Over Scenarios - Expected profit {round(expected_profit)} and its standard deviation {round(standard_deviation_profit)}"
    )
    plt.xlabel("Profit (k€)")
    plt.minorticks_on()
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(
        visible=True,
        which="major",
        linestyle="--",
        dashes=(5, 10),
        color="gray",
        linewidth=0.5,
        alpha=0.8,
    )
    plt.grid(which="minor", visible=False)
    plt.show()

    return expected_profit, standard_deviation_profit


from Step1.CVaRModels import CVaR_onePriceBalancingScheme as CVaR_OPBS
from Step1.CVaRModels import CVaR_twoPriceBalancingScheme as CVaR_TPBS
from tqdm import tqdm
import sys
import os
def expected_profit_vs_CVaR(
    scenarios: list, 
    alphas: list = [0.95], 
    balancingScheme: str = "two", 
    betas: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
):
    """
    Plot expected profit versus Conditional Value at Risk (CVaR) for different alpha levels.
    
    Parameters:
        scenarios (list): List of scenarios.
        alphas (list, optional): List of alpha levels. Default is [0.95].
        balancingScheme (str, optional): Balancing scheme type. Either "one" or "two". Default is "two".
        betas (list, optional): List of beta levels. Default is [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
        
    Returns:
        None
    
    Note:
        This function plots the expected profit versus CVaR for different alpha levels on the same graph.
        The plot visualizes the relationship between the expected profit and CVaR for different risk levels.
    """
    for alpha in alphas:
        print(f"Working on alpha: {alpha}")
        
        CVaRs = []
        expected_profits = []
        sys.stdout = open(os.devnull, 'w')

        for beta in tqdm(betas):
            if balancingScheme == "one":
                model = CVaR_OPBS(alpha=alpha, beta=beta, scenarios=scenarios)
            else:
                 model, obj_initial, production_DA, delta, delta_up, delta_down, eta, zeta= CVaR_TPBS(
                    alpha=alpha, beta=beta, scenarios=scenarios
                ) 
            CVaR = compute_CVaR(
                scenarios=scenarios,
                model=model,
                alpha=alpha,
                balancingScheme=balancingScheme,
                production_DA_dicc=production_DA,
                delta_dicc=delta,
                delta_up_dicc=delta_up,
                delta_down_dicc=delta_down,
                eta_dicc=eta,
                zeta=zeta,
            )
            profits = compute_profits(
                scenarios=scenarios,
                m=model,
                balancingScheme=balancingScheme,
                production_DA_dicc=production_DA,
                delta_dicc=delta,
                delta_up_dicc=delta_up,
                delta_down_dicc=delta_down,
            )
            expected_profit = np.mean(profits)
            expected_profits.append(expected_profit)
            CVaRs.append(CVaR)
        sys.stdout = sys.__stdout__

        expected_profits = np.array(expected_profits)
        CVaRs = np.array(CVaRs)

        plt.plot(CVaRs / 10**3, expected_profits / 10**3, label=rf"$\alpha={alpha}$")
        plt.scatter(CVaRs / 10**3, expected_profits / 10**3, marker="*")
        for i, txt in enumerate(betas):
            plt.annotate(txt, 
                        xy = (CVaRs[i]/10**3, expected_profits[i]/10**3+0.01), 
                        ha = 'center',
                        )

    plt.legend()
    plt.grid(visible=True,which="major",linestyle="--",dashes=(5, 10),color="gray",linewidth=0.5,alpha=0.8,)
    plt.title(f"Expected profit versus CVaR for {balancingScheme} price scheme")
    plt.ylabel("Expected Profit [k€]")
    plt.xlabel("CVaR [k€]")
    plt.minorticks_on()
    plt.show()




