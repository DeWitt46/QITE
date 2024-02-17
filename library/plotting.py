# -*- coding: utf-8 -*-
r"""
Collection of functions to plot the QMETTS_results.



Created on Tue Dec 28 09:34:12 2023

@author: DeWitt
"""
from qiskit.quantum_info import Statevector
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


def plot_qite(QMETTS_result, state_label_list=None):
    color_list = {
        "0--0": "blue",
        "0--1": "red",
        "0-+0": "green",
        "0-+1": "orange",
        "0+-0": "black",
        "0+-1": "darkorange",
        "0++0": "violet",
        "0++1": "purple",
        "1--0": "yellow",
        "1--1": "gold",
        "1-+0": "dodgerblue",
        "1-+1": "powderblue",
        "1+-0": "cyan",
        "1+-1": "magenta",
        "1++0": "crimson",
        "1++1": "pink",
        "-00-": "aquamarine",
        "-00+": "lightseagreen",
        "-01-": "seagreen",
        "-01+": "lime",
        "-10-": "slategray",
        "-10+": "navy",
        "-11-": "slateblue",
        "-11+": "rebeccapurple",
        "+00-": "plum",
        "+00+": "peru",
        "+01-": "firebrick",
        "+01+": "gainsboro",
        "+10-": "rosybrown",
        "+10+": "gray",
        "+11-": "tan",
        "+11+": "greenyellow",
    }

    # color_list = {
    #     "0-0": "blue",
    #     "0-1": "red",
    #     "0+0": "green",
    #     "0+1": "orange",
    #     "1-0": "black",
    #     "1-1": "violet",
    #     "1+0": "purple",
    #     "1+1": "yellow",
    #     "-0-": "gold",
    #     "-0+": "dodgerblue",
    #     "-1-": "powderblue",
    #     "-1+": "cyan",
    #     "+0-": "magenta",
    #     "+0+": "crimson",
    #     "+1-": "pink",
    #     "+1+": "darkorange",
    # }
    # color_list = {
    #     "00": "blue",
    #     "01": "red",
    #     "10": "green",
    #     "11": "orange",
    #     "0-": "black",
    #     "1-": "violet",
    #     "0+": "purple",
    #     "1+": "yellow",
    #     "-0": "gold",
    #     "-1": "dodgerblue",
    #     "+0": "powderblue",
    #     "+1": "cyan",
    #     "--": "magenta",
    #     "-+": "crimson",
    #     "+-": "pink",
    #     "++": "darkorange",
    # }
    taus, evolved_state_dict = QMETTS_result.get_qite()

    N = QMETTS_result.N
    gy = QMETTS_result.gy
    B = QMETTS_result.B
    H = LMG_hamiltonian(N, gy, B)
    ground_state_list = [1.0 for i in range(len(taus))]
    exact_ground_state = Statevector(H.get_ground_state()[1])

    plt.rcParams["figure.figsize"] = [7.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    gylabel = "$\gamma$ = {gy:.1f}, B = {B:.1f}"

    plt.title(
        "QITE and numerical ground state overlap for N = {N:.0f} spins".format(
            N=N
        )
    )

    plt.plot(
        taus,
        ground_state_list,
        color="black",
        ls="dotted",
        label=gylabel.format(gy=gy, B=B),
    )
    if state_label_list is None:
        state_label_list = evolved_state_dict.keys()
    for state_label in state_label_list:
        overlap = []
        for tau_index in range(len(taus)):
            overlap.append(
                np.absolute(
                    evolved_state_dict[state_label][tau_index].inner(
                        exact_ground_state
                    )
                )
            )
        plt.plot(
            taus,
            overlap,
            color=color_list[state_label],
            label="Initial state: {}".format(state_label),
        )
        print(state_label, evolved_state_dict[state_label][-1])
    print("exact_ground_state = ", exact_ground_state)
    plt.ylabel("Overlap")
    plt.xlabel("tau")

    # Line of codes to avoid repeating labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left")

    plt.grid()
    plt.show()
