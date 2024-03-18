# -*- coding: utf-8 -*-
r"""
Class to define the structure of the QMETTS and MHETS algorithms instances results.


Created on Tue Dec 14 12:04:17 2023

@author: DeWitt
"""
from qiskit.quantum_info import Statevector


class QMETTS_results:
    r"""Structure that handles the QMEETS instances results.

    The class takes the different lists and dictionary made by the QMETTS algorithm code which contain fragmented information and organizes all in such a way you can plot what you need.

    Args:
        N: Number of Qubits.
        gy: :math:`\gamma` of the LMG Hamiltonian.
        B: Magnetic field intensity.
        multi_beta_qmetts_result: List of results made by the function qmetts of the QMETTS_instance class.
        preparation_result: Dictionary which contains the evolved basis statevectors.
        preparation_exp_value: Dictionary which contains the expectation values of the observable on the evolved basis statevectors.
    """

    def __init__(
        self,
        N: int,
        gy: float,
        B: float,
        multi_beta_qmetts_result: list,
        preparation_result,
        preparation_exp_values,
    ):
        self.N = N
        self.gy = gy
        self.B = B
        self.multi_beta_qmetts_result = multi_beta_qmetts_result
        self.preparation_result = preparation_result
        self.preparation_exp_values = preparation_exp_values

    def get_total_state_list(self):
        r"""Returns the Markov chain for each beta.

        Returns:
            List of lists sorted by beta. Each internal list contains the Markov chain for that beta.
        """
        total_state_list = []
        for beta_index in range(len(self.multi_beta_qmetts_result)):
            total_state_list.append(
                self.multi_beta_qmetts_result[beta_index]["state_list"]
            )
        return total_state_list

    def get_thermal_averages(self):
        r"""Returns the thermal averages for each beta.

        Returns:
            List of thermal averages of the observable on the each thermal state.
        """
        th_averages = []
        for beta_index in range(len(self.multi_beta_qmetts_result)):
            th_averages.append(self.multi_beta_qmetts_result[beta_index]["th_average"])
        return th_averages

    def get_beta_list(self):
        r"""Returns the list of betas for :math:`\rho_{\beta}`.

        Returns:
            List of betas. The order you see here is the same of every other beta-ordered list.
        """
        beta_list = []
        for beta_index in range(len(self.multi_beta_qmetts_result)):
            beta_list.append(self.multi_beta_qmetts_result[beta_index]["beta"])
        return beta_list

    def get_qite(self):
        state_zero = [1.0]
        for qbit in range(2 ** self.N - 1):
            state_zero.append(0)
        evolved_state_dict = {}
        for key in self.preparation_result.keys():
            taus = self.preparation_result[key]["time_list"]
            evolved_state_dict[key] = []
            for tau_index in range(len(taus)):
                evolved_state = Statevector(state_zero).evolve(
                    self.preparation_result[key]["circuit_list"][tau_index]
                )
                evolved_state_dict[key].append(evolved_state)
        return taus, evolved_state_dict


# class MHETS_results:
#     r"""Structure that handles the QMEETS instances results.

#     The class takes the different lists and dictionary made by the QMETTS algorithm code which contain fragmented information and organizes all in such a way you can plot what you need.

#     Args:
#         N: Number of Qubits.
#         gy: :math:`\gamma` of the LMG Hamiltonian.
#         B: Magnetic field intensity.
#         multi_beta_qmetts_result: List of results made by the function qmetts of the QMETTS_instance class.
#         preparation_result: Dictionary which contains the evolved basis statevectors.
#         preparation_exp_value: Dictionary which contains the expectation values of the observable on the evolved basis statevectors.
#     """

#     data = {"prova1": 3}
#     path = "./MHETS_data/"
#     file_name = "prova.pickle"
#     with open(path + file_name, "wb") as f:
#         # Pickle the preparation_result dictionary using the highest protocol available.
#         pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
