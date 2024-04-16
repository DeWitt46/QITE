# -*- coding: utf-8 -*-
r"""
Functions to implement SPSA optimizer in scipy fashion



Created on Wed Apr 10 16:34:31 2024

@author: DeWitt
"""
from qiskit_algorithms.optimizers import SPSA


def spsa_optimization(
    cost, parameters, args, maxiter=300, learning_rate=None, perturbation=None
):
    def lambda_cost(cost, args):
        return lambda parameters: cost(parameters, *args)

    spsa = SPSA(maxiter=maxiter, learning_rate=learning_rate, perturbation=perturbation)

    return spsa.minimize(fun=lambda_cost(cost, args), x0=parameters)
