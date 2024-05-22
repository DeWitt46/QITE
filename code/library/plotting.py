# -*- coding: utf-8 -*-
r"""
Collection of functions to plot the QMETTS_results.



Created on Tue Dec 28 09:34:12 2023

@author: DeWitt
"""
from qiskit.quantum_info import (
    Statevector,
    state_fidelity,
    partial_trace,
    SparsePauliOp,
)
import numpy as np
import matplotlib.pyplot as plt


from library.operator_creation import LMG_hamiltonian


def plot_thermal_average(QMETTS_result, numerical_final_beta=7.0, numerical_beta_points=100):
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
        num_th_averages.append(LMG_hamiltonian.thermal_average(H, op=H.get_matrix(), beta=beta))
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

    plt.title("Histogram of the Markov chain for N = {N:.0f} spins".format(N=N))

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
    # color_list = {
    #     "0--0": "blue",
    #     "0--1": "red",
    #     "0-+0": "green",
    #     "0-+1": "orange",
    #     "0+-0": "black",
    #     "0+-1": "darkorange",
    #     "0++0": "violet",
    #     "0++1": "purple",
    #     "1--0": "yellow",
    #     "1--1": "gold",
    #     "1-+0": "dodgerblue",
    #     "1-+1": "powderblue",
    #     "1+-0": "cyan",
    #     "1+-1": "magenta",
    #     "1++0": "crimson",
    #     "1++1": "pink",
    #     "-00-": "aquamarine",
    #     "-00+": "lightseagreen",
    #     "-01-": "seagreen",
    #     "-01+": "lime",
    #     "-10-": "slategray",
    #     "-10+": "navy",
    #     "-11-": "slateblue",
    #     "-11+": "rebeccapurple",
    #     "+00-": "plum",
    #     "+00+": "peru",
    #     "+01-": "firebrick",
    #     "+01+": "gainsboro",
    #     "+10-": "rosybrown",
    #     "+10+": "gray",
    #     "+11-": "tan",
    #     "+11+": "greenyellow",
    # }

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
    color_list = {
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
        "--": "magenta",
        "-+": "pink",
        "+-": "crimson",
        "++": "darkorange",
    }
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

    plt.title("QITE and numerical ground state overlap for N = {N:.0f} spins".format(N=N))

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
                np.absolute(evolved_state_dict[state_label][tau_index].inner(exact_ground_state))
            )
        plt.plot(
            taus,
            overlap,
            color=color_list[state_label],
            label="Initial state: {}".format(state_label),
            lw=2.5,
        )
        print(state_label, evolved_state_dict[state_label][-1])
    print("exact_ground_state = ", exact_ground_state)
    plt.ylabel("Overlap")
    plt.xlabel("tau")

    # Line of codes to avoid repeating labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="center right")

    plt.grid()
    plt.show()


def plot_fidelity(beta_list, rho_s_list, H, backend=None):
    N = H.N
    gy = H.gy
    B = H.B
    plt.rcParams["figure.figsize"] = [7.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    plt.title("Fidelity of MHETS and numerical thermal state for N = {N:.0f} spins".format(N=N))
    plt.ylabel("Fidelity")
    plt.xlabel("beta")

    fidelity = []
    for index in range(len(beta_list)):
        fidelity.append(state_fidelity(rho_s_list[index], H.get_thermal_state(beta_list[index])))
    plt.plot(beta_list, np.ones(len(beta_list)), color="black", ls="dotted")
    if backend is None:
        gylabel = "$\gamma$ = {gy:.1f}, B = {B:.1f}"
        plt.scatter(
            beta_list,
            fidelity,
            color="red",
            ls="dotted",
            label=gylabel.format(gy=gy, B=B),
        )
    else:
        gylabel = "$\gamma$ = {gy:.1f}, B = {B:.1f} - MHETS on {backend} backend"
        plt.scatter(
            beta_list,
            fidelity,
            color="red",
            ls="dotted",
            label=gylabel.format(gy=gy, B=B, backend=backend),
        )
    # Line of codes to avoid repeating labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid()
    plt.show()


def plot_rel_entropy(beta_list, rho_s_list, H, backend=None):
    N = H.N
    gy = H.gy
    B = H.B
    plt.rcParams["figure.figsize"] = [7.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    plt.title(
        "Relative entropy of MHETS over numerical thermal state for N = {N:.0f} spins".format(
            N=N
        )
    )
    plt.ylabel("S(MHETS_state||num_state)")
    plt.xlabel("beta")

    rel_entropy = []
    for index in range(len(beta_list)):
        rel_entropy.append(
            H.relative_entropy(
                rho_s_list[index].to_operator().to_matrix(),
                H.get_thermal_state(beta_list[index]),
            )
        )
    plt.plot(beta_list, np.zeros(len(beta_list)), color="black", ls="dotted")
    if backend is None:
        gylabel = "$\gamma$ = {gy:.1f}, B = {B:.1f}"
        plt.scatter(
            beta_list,
            rel_entropy,
            color="red",
            ls="dotted",
            label=gylabel.format(gy=gy, B=B),
        )
    else:
        gylabel = "$\gamma$ = {gy:.1f}, B = {B:.1f} - MHETS on {backend} backend"
        plt.scatter(
            beta_list,
            rel_entropy,
            color="red",
            ls="dotted",
            label=gylabel.format(gy=gy, B=B, backend=backend),
        )
    # Line of codes to avoid repeating labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid()
    plt.show()


def plot_parity(
    beta_list,
    rho_s_list,
    H,
    backend=None,
    numerical_final_beta=7.0,
    numerical_beta_points=100,
):
    N = H.N
    gy = H.gy
    B = H.B
    num_beta = np.linspace(0.2, numerical_final_beta, numerical_beta_points)
    plt.rcParams["figure.figsize"] = [7.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    plt.title("Parity of MHETS over numerical thermal state for N = {N:.0f} spins".format(N=N))
    plt.ylabel("Tr[P rho]")
    plt.xlabel("beta")

    num_parity = []
    P = SparsePauliOp(N * "Z", 1.0)
    for beta in num_beta:
        num_parity.append(H.thermal_average(P.to_matrix(), beta))
    MHETS_parity = []
    for index in range(len(beta_list)):
        MHETS_parity.append(rho_s_list[index].expectation_value(P.to_operator()))
    plt.plot(
        num_beta,
        num_parity,
        color="red",
        ls="dotted",
        label="Numerical",
    )
    if backend is None:
        gylabel = "MHETS - $\gamma$ = {gy:.1f}, B = {B:.1f}"
        plt.scatter(
            beta_list,
            MHETS_parity,
            color="red",
            ls="dotted",
            label=gylabel.format(gy=gy, B=B),
        )
    else:
        gylabel = "MHETS - $\gamma$ = {gy:.1f}, B = {B:.1f} - MHETS on {backend} backend"
        plt.scatter(
            beta_list,
            MHETS_parity,
            color="red",
            ls="dotted",
            label=gylabel.format(gy=gy, B=B, backend=backend),
        )
    # Line of codes to avoid repeating labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid()
    plt.show()


def plot_MHETS_thermal_average(
    beta_list, rho_s_list, H, beta_start=0.1, beta_final=6.0, num_beta=150, backend=None
):
    N = H.N
    gy = H.gy
    B = H.B

    betas = np.linspace(beta_start, beta_final, num_beta)
    thermal_averages = []
    for rho_s in rho_s_list:
        thermal_averages.append(np.trace(rho_s @ H.get_matrix()))
    ground_state_energy = []
    num_th_averages = []
    for beta in betas:
        ground_state_energy.append(H.get_ground_state()[0])
        num_th_averages.append(LMG_hamiltonian.thermal_average(H, op=H.get_matrix(), beta=beta))
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
    if backend is None:
        plt.scatter(
            np.array(beta_list),
            thermal_averages,
            color="red",
            label="MHETS average",
        )
    else:
        plt.scatter(
            np.array(beta_list),
            thermal_averages,
            color="red",
            label="MHETS average on {} backend".format(backend),
        )
    plt.ylabel("Thermal averages")
    plt.xlabel("beta")

    # Line of codes to avoid repeating labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid()
    plt.show()


def plot_MHETS_shots(multi_data, betas, H):
    color_list = ["blue", "red", "green", "gold", "orange"]

    fig, axs = plt.subplots(len(betas), sharex=True)
    plt.rcParams["figure.figsize"] = [7.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    gylabel = "QASM with n_shots = {}"

    fig.suptitle(
        "Helmoltz energy optimization for N = {N:.0f}, $\gamma$ = {gy:.1f}, B = {B:.1f}".format(
            N=H.N, gy=H.gy, B=H.B
        )
    )

    for beta in betas:
        maxiter_list = range(1, multi_data[0]["optimization_options"]["maxiter"] + 1)
        axs[np.where(betas == beta)[0][0]].plot(
            maxiter_list,
            [H.cost_function(beta) / beta for _ in range(len(maxiter_list))],
            color="black",
            ls="dotted",
            label="Numerical value for beta = {}".format(beta),
        )

        for index in range(len(multi_data)):
            beta_index = multi_data[index]["betas"].index(beta)
            n_iterations = multi_data[index]["n_eval"][beta_index] + 1
            if multi_data[index]["optimization_options"]["optimizer"] in (
                "spsa",
                "SPSA",
            ):
                n_iterations += 51  # SPSA takes some iterations for calibration
            axs[np.where(betas == beta)[0][0]].plot(
                range(1, n_iterations),
                take_helm_energy(multi_data[index], beta, H),
                color=color_list[index],
                label=gylabel.format(multi_data[index]["optimization_options"]["shots"]),
            )
    for ax in axs.flat:
        ax.set(xlabel="N evaluation", ylabel="Helmoltz energy")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
        ax.legend()  # Show legend


def plot_MHETS_shots_fidelity(multi_data, betas, H):
    color_list = ["blue", "red", "green", "gold", "orange"]

    fig, axs = plt.subplots(len(betas), sharex=True)
    plt.rcParams["figure.figsize"] = [7.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    gylabel = "QASM with n_shots = {}"

    fig.suptitle(
        "QST fidelity during optimization for N = {N:.0f}, $\gamma$ = {gy:.1f}, B = {B:.1f}".format(
            N=H.N, gy=H.gy, B=H.B
        )
    )

    for beta in betas:
        maxiter_list = range(1, multi_data[0]["optimization_options"]["maxiter"] + 1)
        axs[np.where(betas == beta)[0][0]].plot(
            maxiter_list,
            [1.0 for _ in range(len(maxiter_list))],
            color="black",
            ls="dotted",
            label="Max value - beta = {}".format(beta),
        )

        for index in range(len(multi_data)):
            beta_index = multi_data[index]["betas"].index(beta)
            axs[np.where(betas == beta)[0][0]].plot(
                range(1, multi_data[index]["n_eval"][beta_index] + 1),
                take_QST_fidelity(multi_data[index], beta, H),
                color=color_list[index],
                label=gylabel.format(multi_data[index]["optimization_options"]["shots"]),
            )
    for ax in axs.flat:
        ax.set(xlabel="N evaluation", ylabel="QST Fidelity")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
        ax.legend()  # Show legend


def take_QST_fidelity(multi_beta_result, beta, H):
    QST_fidelity_list = []

    beta_index = multi_beta_result["betas"].index(beta)
    for data in multi_beta_result["callback_data"][beta_index]:
        QST_fidelity_list.append(data.analysis_results("state_fidelity").value)
    return QST_fidelity_list


def take_helm_energy(multi_beta_result, beta, H):
    helm_energy_list = []

    beta_index = multi_beta_result["betas"].index(beta)
    for data in multi_beta_result["callback_data"][beta_index]:
        rho = data.analysis_results("state").value
        prob_ancilla = rho.probabilities(range(0, H.N))

        entropy = 0.0
        for index in range(len(prob_ancilla)):
            if prob_ancilla[index] != 0.0:
                entropy -= prob_ancilla[index] * np.log(prob_ancilla[index])
        rho_S = partial_trace(rho, range(H.N))

        system_exp_value = np.trace(rho_S @ H.get_matrix())

        helm_energy_list.append(np.real(system_exp_value - (1.0 / beta) * entropy))
    return helm_energy_list
