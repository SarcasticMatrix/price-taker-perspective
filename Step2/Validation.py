def validation_out_of_sample(
        scenarios: list, opt_reserve_cap_bid: float):
    """
    Validation in-sample decision making

    Inputs:
    - scenarios (list of pd.dataframe): list of all the 150 scenarios (out of sample)
    - opt_reserve_cap_bid: Result of the optimization problem (In-Sample analysis)

    Outputs:
    - exp_reserve_shortfall: float
    - Valitated: bool
    - i: index of the non respective criteria one in case of a fault
    """
    nbMin=60
    nbSamples=len(scenarios)
    F_up = np.array(
        [scenarios[i].values for i in range(nbSamples)]
    )

    #reserve shortfall
    opt_mat=np.array(
        [opt_reserve_cap_bid for i in range(nbSamples)]
    )

    Delta = F_up - opt_mat
    exp_reserve_shortfall= - min(Delta)

    #Validation:
    Valitated = True
    i=0
    while (i < nbSamples) and Valitated == True:
        count=0
        list = scenarios[i].values
        for j in range(nbMin):
            if list[j]<opt_reserve_cap_bid:
                count+=1
        if count>nbMin/10:
            Valitated = False
        i+=1

    return  exp_reserve_shortfall, Valitated, i