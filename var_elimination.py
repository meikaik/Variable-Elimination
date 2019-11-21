import pandas as pd
import sys
import json


def restrict(factor, variable, value):
    """
	Restricts a variable to some value in a given factor
	"""
    factor = factor.copy()
    return factor[factor[variable] == value].drop(columns=variable)


def sumout(factor, variable):
    """
    Sums out a variable given a factor
    """
    factor = factor.copy()
    factor.pop(variable)
    vars = list(factor.columns)[0:-1]
    if len(vars) == 0:
        return pd.DataFrame({"Prob": [factor["Prob"].sum()]})
    else:
        return factor.groupby(vars, as_index=False).aggregate({"Prob": "sum"})


def multiply(factor1, factor2):
    """
    Multiplies two factors (pointwise multiplication)
    """
    factor1 = factor1.copy()
    factor2 = factor2.copy()
    var_1 = list(factor1.columns)[0:-1]
    var_2 = list(factor2.columns)[0:-1]

    # XOR check for var1 and var2
    if bool(var_1) ^ bool(var_2):
        if len(var_2) == 0:
            factor1["Prob"] = factor1["Prob"].apply(
                lambda x: x * factor2["Prob"].values[0]
            )
            return factor1
        factor2["Prob"] = factor2["Prob"].apply(lambda x: x * factor1["Prob"].values[0])
        return factor2

    intersection = list(set(var_1) & set(var_2))
    if len(intersection) == 0:
        raise Exception("Implement no intersection case", var_1, var_2)

    factor3 = factor1.merge(factor2, on=intersection)
    factor3["Prob"] = factor3["Prob_x"] * factor3["Prob_y"]
    factor3.pop("Prob_x")
    factor3.pop("Prob_y")
    return factor3


def normalize(factor):
    """
    Divides each entry by the sum of all entries
    """
    factor = factor.copy()
    sum = factor["Prob"].sum()
    factor["Prob"] = factor["Prob"] / sum
    return factor


def sumout_print(factor, variable, silent=True):
    if silent:
        return sumout(factor, variable)
    print("Sumout " + variable + " from:")
    print(factor)
    factor = sumout(factor, variable)
    print("\nResult:")
    print(factor)
    print()
    return factor


def restrict_print(factor, variable, value, silent=True):
    if silent:
        return restrict(factor, variable, value)
    print("Restricted to " + variable + " = " + str(value) + "\n")
    print(factor)
    factor = restrict(factor, variable, value)
    print("\nResult:")
    print(factor)
    print()
    return factor


def multiply_print(factor1, factor2, silent=True):
    if silent:
        return multiply(factor1, factor2)
    print("Multiplying:")
    print(factor1)
    print(factor2)
    factor3 = multiply(factor1, factor2)
    print("\nResult:")
    print(factor3)
    print()
    return factor3


def normalize_print(factor, silent=True):
    if silent:
        return normalize(factor)
    print("Normalize:")
    print(factor)
    print("\nResult:")
    factor = normalize(factor)
    print(factor)
    print()
    return factor


def inference(factor_list, query_variables, evidence_list=[], hidden_vars=None, silent=True):
    if hidden_vars is None:
        hidden_vars = list(set(variables()) - set(query_variables) - set(evidence_list))

    # Restrict the observed variables to their observed values based on evidence
    for evidence in evidence_list:
        for i, factor in enumerate(factor_list):
            if evidence in factor.columns:
                factor_list[i] = restrict_print(factor, evidence, True, silent)

    # Eliminate each hidden variable Xhj
    for hidden_var in hidden_vars:
        to_eliminate = []
        for i, factor in reversed(list(enumerate(factor_list))):
            if hidden_var in factor.columns:
                to_eliminate.append(factor_list.pop(i))

        if len(to_eliminate) != 0:
            new_factor = to_eliminate.pop(0)
            # Multiply all the factors that contain Xhj to get new factor gj.
            for factor in to_eliminate:
                new_factor = multiply_print(new_factor, factor, silent)

            # Sum out the variable Xhj from the factor gj.
            if hidden_var in new_factor.columns:
                new_factor = sumout_print(new_factor, hidden_var, silent)
            factor_list.append(new_factor)

    resulting_factor = factor_list.pop(0)
    # Multiply the remaining factors
    for factor in factor_list:
        resulting_factor = multiply_print(resulting_factor, factor, silent)

    # Normalize the resulting factor
    return normalize_print(resulting_factor, silent)


def factor_list():
    with open("input.json") as json_file:
        data = json.load(json_file)
        dfFH = pd.DataFrame(data["prFH"])
        dfFS = pd.DataFrame(data["prFS"])
        dfFB = pd.DataFrame(data["prFB"])
        dfFM = pd.DataFrame(data["prFM"])
        dfNA = pd.DataFrame(data["prNA"])
        dfNDG = pd.DataFrame(data["prNDG"])
        return [dfFB, dfFH, dfFS, dfFM, dfNA, dfNDG]


def variables():
    with open("input.json") as json_file:
        data = json.load(json_file)
        return data["variables"]


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 var_elimination.py <a|b|c|d>")
        return
    q = sys.argv[1]
    if q == "a":
        # Pr(FH):
        inference(factor_list(), ["FH"], silent=False)
    elif q == "b":
        # Pr(FS | FM & FH)
        inference(factor_list(), ["FS"], ["FM", "FH"], silent=False)
    elif q == "c":
        # Pr(FS | FM & FH & FB)
        inference(factor_list(), ["FS"], ["FM", "FH", "FB"], silent=False)
    elif q == "d":
        # Pr(FS | FH & FM & FB & NA)
        inference(factor_list(), ["FS"], ["FH", "FM", "FB", "NA"], silent=False)
    else:
        print("Usage: python3 var_elimination.py <a|b|c|d>")


if __name__ == "__main__":
    main()
