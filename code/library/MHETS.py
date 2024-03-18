# -*- coding: utf-8 -*-
r"""
Class to create an instance of the MHETS (Minimize Helmoltz Energy for Thermal States) algorithm.



Created on Fri Feb 23 22:57:31 2024

@author: DeWitt
"""
import pickle
import numpy as np
import time


from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit_experiments.library import StateTomography

from library import montecarlo
from library.operator_creation import LMG_hamiltonian
from library.ansatz_creation import two_local


class MHETS_instance:
    """
    Creates an instance of the MHETS algorithm.
    Variational algorithm that minimizes Helmoltz free energy to generate thermal state, proposed in arXiv:2303.11276.
    """

    def __init__(
        self,
        H: LMG_hamiltonian,
        ancilla_ansatz: two_local,
        system_ansatz: two_local,
        optimization_options: dict,
        flag="statevector",
        backend=None,
        initial_parameter_list=None,
    ):
        self.H = H
        self.N = H.N
        self.gy = H.gy
        self.B = H.B
        self.ancilla_ansatz = ancilla_ansatz
        self.N_ancilla = ancilla_ansatz.num_qubits
        self.system_ansatz = system_ansatz
        self.N_system = system_ansatz.num_qubits
        if self.N_ancilla != self.N_system:
            print("Warning!!! N ancilla != N system")
        if self.N != self.N_system:
            print("Warning!!! N Hamiltonian != N system ansatz")
        if ancilla_ansatz.get_par_name() == system_ansatz.get_par_name():
            self.system_ansatz.par_name = "{}{}".format(
                system_ansatz.get_par_name(), system_ansatz.get_par_name()
            )
            print(
                "Changed system_ansatz parameters name to {}".format(
                    self.system_ansatz.get_par_name()
                )
            )
        self.initial_parameter_list = initial_parameter_list
        if self.initial_parameter_list == None:
            self.initial_parameter_list = np.zeros(
                self.ancilla_ansatz.get_num_parameters()
                + self.system_ansatz.get_num_parameters()
            )
        elif len(self.initial_parameter_list) != (
            self.ancilla_ansatz.get_num_parameters()
            + self.system_ansatz.get_num_parameters()
        ):
            print(
                "Warning!!! Initial parameter list number not correct. Changed to [0, ..., 0]"
            )
            self.initial_parameter_list = np.zeros(
                self.ancilla_ansatz.get_num_parameters()
                + self.system_ansatz.get_num_parameters()
            )
        self.current_parameter_list = self.initial_parameter_list
        self.flag = flag
        self.backend = backend
        self.optimization_options = optimization_options

    def get_N(self):
        return self.N

    def get_N_ancilla(self):
        return self.N_ancilla

    def get_current_parameters(self):
        return self.current_parameter_list

    def build_total_circuit(self):
        ancilla_qc = self.ancilla_ansatz.build()
        system_qc = self.system_ansatz.build()
        total_qc = QuantumCircuit(self.N_ancilla + self.N)
        total_qc.append(ancilla_qc, range(0, self.N_ancilla))
        for qbit in range(0, self.N):
            total_qc.cx(qbit, qbit + self.N_ancilla)
        total_qc.append(system_qc, range(self.N_ancilla, self.N_ancilla + self.N))
        return total_qc

    def draw_total_circuit(self):
        print(self.build_total_circuit().decompose().draw())
        return

    def update_parameters(self, parameter_list: list):
        self.current_parameter_list = parameter_list
        ancilla_parameters = [
            parameter_list[i]
            for i in range(0, self.ancilla_ansatz.get_num_parameters())
        ]
        system_parameters = [
            parameter_list[i]
            for i in range(
                self.ancilla_ansatz.get_num_parameters(),
                self.ancilla_ansatz.get_num_parameters()
                + self.system_ansatz.get_num_parameters(),
            )
        ]
        self.ancilla_ansatz.bind_parameters(ancilla_parameters)
        self.system_ansatz.bind_parameters(system_parameters)

    def QST(self, circuit):
        if self.flag == "statevector":
            rho = DensityMatrix(circuit)
            return rho
        elif self.flag == "noise":
            if self.backend is None:
                print("You didn't indicate backend")
            experiment = StateTomography(circuit)
            data = experiment.run(self.backend).block_for_results()

            return data

    def cost_function(self, parameter_list: list, beta):
        # print("updating parameter")
        self.update_parameters(parameter_list)

        start = time.time()
        total_qc = self.build_total_circuit()
        if self.backend is not None:
            data = self.QST(total_qc)
            rho = data.analysis_results("state").value
        else:
            rho = self.QST(total_qc)
        finish = time.time()
        # print("Time rho building =", finish - start)
        prob_ancilla = rho.probabilities(range(0, self.N_ancilla))

        entropy = 0.0
        for index in range(len(prob_ancilla)):
            if prob_ancilla[index] != 0.0:
                entropy -= prob_ancilla[index] * np.log(prob_ancilla[index])
        # system_exp_value = rho.expectation_value(
        #     self.H.get_pauli(), range(self.N_ancilla, self.N_ancilla + self.N)
        # ) # Old method, time consuming
        rho_S = partial_trace(rho, range(self.N_ancilla))

        system_exp_value = np.trace(rho_S @ self.H.get_matrix())
        # print("Time energy exp_value measured =", time.time() - finish)

        # if np.imag(system_exp_value) != 0.0:
        #     print("Warning!!! Exp_value Imag = ", np.imag(system_exp_value))
        # print("Current F =", np.real(beta * system_exp_value - entropy))
        return np.real(beta * system_exp_value - entropy)

    def optimize(
        self, beta, initial_parameter_list=None, optimizer="SLSQP", maxiter=1000,
    ):
        total_start = time.time()
        if initial_parameter_list is None:
            scipy_result = minimize(
                self.cost_function,
                self.initial_parameter_list,
                args=(beta),
                method=optimizer,
                options={"maxiter": maxiter},
            )
        else:
            scipy_result = minimize(
                self.cost_function,
                initial_parameter_list,
                args=(beta),
                method=optimizer,
                options={"maxiter": maxiter},
            )
        print("Total time", time.time() - total_start)
        result = {
            "optimized_parameter_list": scipy_result.x,
            "Helmoltz energy": scipy_result.fun,
            "success": scipy_result.success,
            "Message": scipy_result.message,
            "Time optimization": time.time() - total_start,
            "n_eval": scipy_result.nfev,
        }
        return result

    def multi_beta_optimization_from_scratch(self, betas):
        # INITIALIZE THE DICTIONARY RESULT
        result = self.optimize(
            beta=betas[0],
            initial_parameter_list=self.optimization_options["initial_parameter_list"][
                0
            ],
            maxiter=self.optimization_options["maxiter"],
            optimizer=self.optimization_options["optimizer"],
        )
        multi_beta_result = {
            "betas": [betas[0]],
            "optimization_options": self.optimization_options,
            "backend": self.backend,
        }
        for key in result.keys():
            multi_beta_result[key] = [result[key]]
        # CONTINUE FILLING IT
        for index in range(1, len(betas)):
            result = self.optimize(
                beta=betas[index],
                initial_parameter_list=self.optimization_options[
                    "initial_parameter_list"
                ][index],
                maxiter=self.optimization_options["maxiter"],
                optimizer=self.optimization_options["optimizer"],
            )
            multi_beta_result["betas"].append(betas[index])
            for key in result.keys():
                multi_beta_result[key].append(result[key])
        return multi_beta_result
