# -*- coding: utf-8 -*-
r"""
List of functions useful for MHETS algorithm runs


Created on Wed Feb 28 18:55:30 2024

@author: DeWitt
"""
import numpy as np


# from library.MHETS import MHETS_instance


def sort_result(total_result):
    index_min = np.argmin(total_result["Helmoltz energy"])
    minimized_result = {}
    for key in total_result.keys():
        minimized_result[key] = total_result[key][index_min]
    return minimized_result


def run(
    instance,
    beta,
    n_starting_point=10,
    optimizer="COBYLA",
    maxiter=1000,
    tol=1e-6,
    shots=1024,
):
    starting_point_list = [instance.initial_parameter_list]
    for i in range(n_starting_point - 1):
        starting_point_list.append(
            np.random.rand(len(instance.initial_parameter_list)) * 2 * np.pi - np.pi
        )  # random array of parameters in [-pi, pi]
    print("ccccccccccccccccccccccccccc")
    print("Run for beta =", beta)
    print("ccccccccccccccccccccccccccc")
    total_result = {"Starting point": starting_point_list}
    result = instance.optimize(
        beta,
        initial_parameter_list_guess=starting_point_list[0],
        optimizer=optimizer,
        maxiter=maxiter,
        tol=tol,
        shots=shots,
    )
    for key in result.keys():
        total_result[key] = [result[key]]
    print("ccccccccccccccccccccccccccc")
    print("Initial starting point done")
    print("ccccccccccccccccccccccccccc")

    for parameter_list in starting_point_list[1:]:
        result = instance.optimize(
            beta,
            initial_parameter_list_guess=parameter_list,
            optimizer=optimizer,
            maxiter=maxiter,
            tol=tol,
            shots=shots,
        )
        for key in result.keys():
            total_result[key].append(result[key])
        print("ccccccccccccccccccccccccccc")
        print("Next starting point done")
        print("ccccccccccccccccccccccccccc")
    minimized_result = sort_result(total_result)

    return minimized_result, total_result
