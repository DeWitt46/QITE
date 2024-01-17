# -*- coding: utf-8 -*-
r"""
Collection of functions to plot the QMETTS_results.



Created on Tue Dec 28 09:34:12 2023

@author: DeWitt
"""

import numpy as np
import matplotlib.pyplot as plt


from library.operator_creation import LMG_hamiltonian


def plot_thermal_average(
    QMETTS_result, numerical_final_beta=7.0, numerical_beta_points=100
):
    r"""Plots the energy thermal averages contained in QMETTS_result compared with the numerical ones and the ground state energy.

    Args:
        QMETTS_result: QMETTS_results class containing data you intend to plot.
        numerical_final_beta: Final beta you want to use for the numerical comparison.
        numerical_beta_points: Beta points you want to use for the numerical comparison.
    """
    betas = np.linspace(0.0, numerical_final_beta, numerical_beta_points)
    beta_list = QMETTS_result.get_beta_list()
    thermal_averages = QMETTS_result.get_thermal_averages()

    N = QMETTS_result.N
    gy = QMETTS_result.gy
    B = QMETTS_result.B
    H = LMG_hamiltonian(N, gy, B)
    ground_state_energy = []
    num_th_averages = []
    for beta in betas:
        ground_state_energy.append(H.get_ground_state()[0])
        num_th_averages.append(
            LMG_hamiltonian.thermal_average(H, op=H.get_matrix(), beta=beta)
        )

    plt.rcParams["figure.figsize"] = [7.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    gylabel = "$\gamma$ = {gy:.1f}, B = {B:.1f} - ground state energy"

    plt.title("Energy thermal averages for N = {N:.0f} spins".format(N=N))

    plt.plot(
        betas,
        ground_state_energy,
        color="black",
        ls="dotted",
        label=gylabel.format(gy=gy, B=B),
    )
    plt.plot(betas, num_th_averages, color="blue", label="numerical average")
    plt.scatter(
        np.array(beta_list),
        thermal_averages,
        color="red",
        label="QMETTS average",
    )

    plt.ylabel("Thermal averages")
    plt.xlabel("beta")

    # Line of codes to avoid repeating labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid()
    plt.show()


def plot_state_histogram(QMETTS_result, bins=None):
    r"""Plots the histogram of the Markov chain for the last beta.

    Args:
        QMETTS_result: QMETTS_results class containing data you intend to plot.
        bins: Number of the histogram bins.
    """
    beta = QMETTS_result.get_beta_list()[-1]
    state_list = QMETTS_result.get_total_state_list()[-1]

    N = QMETTS_result.N
    gy = QMETTS_result.gy
    B = QMETTS_result.B

    plt.rcParams["figure.figsize"] = [7.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    gylabel = "$\gamma$ = {gy:.1f}, B = {B:.1f}, 1/T = {beta:.2f}"

    plt.title(
        "Histogram of the Markov chain for N = {N:.0f} spins".format(N=N)
    )

    plt.hist(
        state_list,
        bins=bins,
        color="black",
        label=gylabel.format(gy=gy, B=B, beta=beta),
    )

    plt.ylabel("Counts")
    plt.xlabel("Statevectors label")

    # Line of codes to avoid repeating labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid()
    plt.show()
