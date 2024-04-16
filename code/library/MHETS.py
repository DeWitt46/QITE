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


from library import montecarlo, SPSA_lib
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
        if self.initial_parameter_list is None:
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
        self.current_parameter_list = self.initial_parameter_list.copy()
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

    def QST(self, circuit, shots):
        if self.flag == "statevector":
            rho = DensityMatrix(circuit)
            return rho
        elif self.flag == "qasm":
            if self.backend is None:
                print("You didn't indicate backend")
            experiment = StateTomography(circuit)
            data = experiment.run(self.backend, shots=shots).block_for_results()

            return data
        elif self.flag == "noise":
            if self.backend is None:
                print("You didn't indicate backend")
            experiment = StateTomography(circuit)
            data = experiment.run(self.backend, shots=shots).block_for_results()

            return data

    def cost_function(self, parameter_list: list, beta, shots):
        self.update_parameters(parameter_list)

        global callback_data
        global counter

        total_qc = self.build_total_circuit()
        if self.backend is not None:
            data = self.QST(circuit=total_qc, shots=shots)
            rho = data.analysis_results("state").value
            if counter % 10 == 0:  # Data is heavy, we have to take few
                callback_data.append([counter, data])
        else:
            rho = self.QST(circuit=total_qc, shots=shots)
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
        if self.flag == "statevector":
            callback_data.append(
                [counter, np.real(beta * system_exp_value - entropy)]
            )  # MemoryError for N=4, maxiter about 2-3000 If you want entire rho
        counter += 1
        return np.real(beta * system_exp_value - entropy)

    def optimize(
        self,
        beta,
        initial_parameter_list_guess=None,
        optimizer="SLSQP",
        maxiter=1000,
        tol=1e-1,
        shots=1024,
    ):
        global counter
        global callback_data
        callback_data = []
        counter = 0
        total_start = time.time()
        if initial_parameter_list_guess is None:
            if optimizer == "spsa":
                scipy_result = SPSA_lib.spsa_optimization(
                    cost=self.cost_function,
                    parameters=self.initial_parameter_list,
                    args=(beta, shots),
                    maxiter=maxiter,
                )
            else:
                scipy_result = minimize(
                    self.cost_function,
                    self.initial_parameter_list,
                    args=(beta, shots),
                    method=optimizer,
                    options={"maxiter": maxiter},
                    tol=tol,
                )
        else:
            if optimizer == "spsa":
                scipy_result = SPSA_lib.spsa_optimization(
                    cost=self.cost_function,
                    parameters=initial_parameter_list_guess,
                    args=(beta, shots),
                    maxiter=maxiter,
                )
            else:
                scipy_result = minimize(
                    self.cost_function,
                    initial_parameter_list_guess,
                    args=(beta, shots),
                    method=optimizer,
                    options={"maxiter": maxiter},
                    tol=tol,
                )
        print("Total time", time.time() - total_start)

        result = {
            "optimized_parameter_list": scipy_result.x,
            "Helmoltz energy": scipy_result.fun,
            "Time optimization": time.time() - total_start,
            "n_eval": scipy_result.nfev,
            "callback_data": callback_data,
        }
        return result

    def multi_beta_optimization_from_scratch(self, betas):
        # INITIALIZE THE DICTIONARY RESULT
        result = self.optimize(
            beta=betas[0],
            initial_parameter_list_guess=self.optimization_options[
                "initial_parameter_list"
            ][0],
            maxiter=self.optimization_options["maxiter"],
            optimizer=self.optimization_options["optimizer"],
            tol=self.optimization_options["tol"],
            shots=self.optimization_options["shots"],
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
            # HERE TO CHANGE INITIAL PARAMETER GUESS WITH PREVIOUS OPTIMIZED PARAMETERS
            result = self.optimize(
                beta=betas[index],
                # initial_parameter_list_guess=self.optimization_options[
                #     "initial_parameter_list"
                # ][index], # -> THIS IS STRATEGY A0
                initial_parameter_list_guess=multi_beta_result[
                    "optimized_parameter_list"
                ][
                    index - 1
                ],  # THIS IS STRATEGY A1
                maxiter=self.optimization_options["maxiter"],
                optimizer=self.optimization_options["optimizer"],
                tol=self.optimization_options["tol"],
                shots=self.optimization_options["shots"],
            )
            multi_beta_result["betas"].append(betas[index])
            for key in result.keys():
                multi_beta_result[key].append(result[key])
        return multi_beta_result

    def multi_beta_optimization_from_data(self, betas, old_data):
        # INITIALIZE DICTIONARY RESULT
        multi_beta_result = {
            "optimization_options": self.optimization_options,
            "backend": self.backend,
        }
        for key in old_data.keys():
            if key != "optimization_options":
                if key != "backend":
                    multi_beta_result[key] = []
        # FILLING DICTIONARY RESULT
        for new_beta_index in range(len(betas)):
            if betas[new_beta_index] in old_data["betas"]:
                old_beta_index = old_data["betas"].index(betas[new_beta_index])
                for key in old_data.keys():
                    if key != "optimization_options":
                        if key != "backend":
                            multi_beta_result[key].append(old_data[key][old_beta_index])
            else:
                # TODO: check initial parameters when flag=hardware, not implemented
                result = self.optimize(
                    beta=betas[new_beta_index],
                    # initial_parameter_list_guess=self.optimization_options[
                    #     "initial_parameter_list"
                    # ][new_beta_index], # -> THIS IS STRATEGY A0
                    initial_parameter_list_guess=multi_beta_result[
                        "optimized_parameter_list"
                    ][
                        new_beta_index - 1
                    ],  # THIS IS STRATEGY A1
                    maxiter=self.optimization_options["maxiter"],
                    optimizer=self.optimization_options["optimizer"],
                    tol=self.optimization_options["tol"],
                    shots=self.optimization_options["shots"],
                )
                multi_beta_result["betas"].append(betas[new_beta_index])
                for key in result.keys():
                    multi_beta_result[key].append(result[key])
        return multi_beta_result

    def multi_beta_optimization_run(
        self, betas, n_starting_point=10,
    ):
        # INITIALIZE THE DICTIONARY RESULT
        minimized_result, total_result = montecarlo.run(
            self,
            beta=betas[0],
            n_starting_point=n_starting_point,
            optimizer=self.optimization_options["optimizer"],
            maxiter=self.optimization_options["maxiter"],
            tol=self.optimization_options["tol"],
            shots=self.optimization_options["shots"],
        )
        multi_beta_result = {
            "betas": [betas[0]],
            "optimization_options": self.optimization_options,
            "backend": self.backend,
        }
        multi_beta_runs = [
            {
                "betas": [betas[0]],
                "optimization_options": self.optimization_options,
                "backend": self.backend,
            }
            for _ in range(n_starting_point)
        ]
        for key in minimized_result.keys():
            multi_beta_result[key] = [minimized_result[key]]
        for key in total_result.keys():
            for run_index in range(len(multi_beta_runs)):
                multi_beta_runs[run_index][key] = [total_result[key][run_index]]
        # CONTINUE FILLING IT
        for index in range(1, len(betas)):
            minimized_result, total_result = montecarlo.run(
                self,
                beta=betas[index],
                n_starting_point=n_starting_point,
                optimizer=self.optimization_options["optimizer"],
                maxiter=self.optimization_options["maxiter"],
                tol=self.optimization_options["tol"],
                shots=self.optimization_options["shots"],
            )
            multi_beta_result["betas"].append(betas[index])
            for key in minimized_result.keys():
                multi_beta_result[key].append(minimized_result[key])
            for run_index in range(len(multi_beta_runs)):
                multi_beta_runs[run_index]["betas"].append(betas[index])
            for key in total_result.keys():
                for run_index in range(len(multi_beta_runs)):
                    multi_beta_runs[run_index][key].append(total_result[key][run_index])
        return multi_beta_result, multi_beta_runs
