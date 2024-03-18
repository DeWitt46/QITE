# -*- coding: utf-8 -*-
r"""
Class to create an instance of the QMETTS algorithm.
QMETTS is used to make thermal averages on the state :math:`\rho_{\beta} = \e^{-\beta H}`.
This algorithm evolves each statevector of the basis you indicate to final_beta/2 once and stores all the evolved statevectors for all the intermediate betas.
Then, for each intermediate beta, a random product operator (among the ones you choose) is measured on the evolved basis statevector, gaining a new basis stavector to evolve and so on for the number of shots provided.
Once a shots-long chain of statevectors is determined, for each intermediate beta, the expectation value of the observable is measured and averaged to get the thermal average.
The advantage is that you don't actually evolve a basis stavector to each intermediate  :math:`\beta/2 ` for all of the shots; you just take the result from the initial "evolving basis phase"'s stored results.



Created on Tue Dec 12 16:54:13 2023

@author: DeWitt
"""
import pickle
import numpy as np
import copy


from time_evolution_problem import TimeEvolutionProblem
from QITE.var_qite import VarQITE


from library import state_label as lb
from library import result_handler
from library.ansatz_creation import two_local
from library.operator_creation import LMG_hamiltonian


class QMETTS_instance:
    """
    Creates an instance of the QMETTS algorithm.
    .. code-block::python

        import numpy as np

        from library.ansatz_creation import two_local
        from library.operator_creation import LMG_hamiltonian
        from library.QMETTS import QMETTS_instance
        from qiskit.quantum_info import SparsePauliOp




        hamiltonian = SparsePauliOp.from_list(
            [
                ("II", 0.15),
                ("ZZ", 0.32),
                ("IZ", 0.37),
                ("ZI", 0.31),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )

        # If you want to choose the product operators to measure
        operators = ["xx", "xz"]
        flag = "manual"

        # If you want to take all possible combinations of a certain list of
        # operators to measure
        operators = ["x", "z"] #For N=2-> xx, xz, zx, zz
        flag = "all_combinations"



        final_beta = 0.5
        num_beta_points = 5
        shots = 512
        # Choose an initial state that belongs to the eigenstates of one of
        # the product operators you chose
        initial_state = "++"
        ansatz = twolocal(num_qubits=hamiltonian.num_qubits, reps=1)


        qmetts_instance = QMETTS_instance(
            hamiltonian,
            operators,
            flag,
            final_beta,
            num_beta_points,
            shots,
            initial_state,
            ansatz,
        )

        # Evaluating thermal average
        observable = SparsePauliOp.from_list(
            [
                ("IX", 0.05),
                ("XI", 0.05),
                ("ZZ", 0.2),
            ]
        )
        QMETTS_result = qmetts_instance.multi_beta_qmetts(
            op=observable, initial_state=initial_state, shots=shots
        )
    """

    def __init__(
        self,
        H: LMG_hamiltonian,
        operators: list,
        flag: str,
        beta: float,
        num_beta_points: int,
        shots: int,
        initial_state: str,  # list only for doing QITE
        ansatz: two_local,
        num_timesteps: None = None,
    ):
        self.H = H
        self.N = H.N
        self.gy = H.gy
        self.B = H.B
        self.operators = operators
        self.flag = flag
        self.basis_list = lb.generate_basis_list(
            N=self.N, operators=self.operators, flag=self.flag
        )
        self.basis_measure_list = lb.generate_basis_measure_list(
            N=self.N, operators=self.operators, flag=self.flag
        )
        self.beta = beta
        self.num_beta_points = num_beta_points
        self.shots = shots
        self.initial_state = initial_state
        self.ansatz = ansatz
        self.num_timesteps = num_timesteps
        self.problem = TimeEvolutionProblem(H.get_pauli(), beta)
        if type(initial_state) == list:
            self.qite = VarQITE(
                ansatz.build(), initial_state, num_timesteps=num_timesteps
            )
        elif type(initial_state) == str:
            self.qite = VarQITE(
                ansatz.build(),
                lb.state_to_par(
                    label=initial_state, num_params=ansatz.get_num_parameters()
                ),
                num_timesteps=num_timesteps,
            )
        self.file_name = (
            "preparation_result_{}at_gy{}_B{}_reps{}".format(
                self.N, self.gy, self.B, self.ansatz.num_reps
            ).replace(".", "")
            + ".pickle"
        )

    def get_basis_list(self):
        r"""Gets the basis list you can have when measuring the product operators provided.

        Returns:
            List of all the basis statevectors labels.
        """
        return self.basis_list

    def get_basis_measure_list(self):
        r"""Gets the product operators provided.

        Returns:
            List of all the product operators It measures to get the chain.
        """
        return self.basis_measure_list

    def get_beta_list(self):
        r"""Gets the list of all the betas on which you get the thermal average.

        Returns:
            List of all the betas on which you get the thermal average.
        """
        # 0.01 is the standard way of timestep evolution for VarQITE
        beta_list = np.linspace(0.01, self.beta, self.num_beta_points)
        return beta_list

    def get_tau_list(self):
        r"""Gets the list of all the :math:`\tau = \frac{1}{2}\beta` imaginary time you actually evolve the statevectors.

        Returns:
            List of all the taus you actually evolve the basis statevector.
        """
        # 0.01 is the standard way of timestep evolution for VarQITE
        tau_list = np.linspace(0.01, self.beta / 2.0, self.num_beta_points)
        return tau_list

    def return_tau_index(self, tau):
        r"""Takes a value of tau and returns Its index in the tau list.

        Args:
            tau: The value of tau.

        Returns:
            Index of the tau list corresponding to the value of tau.
        """
        # 0.01 is the standard way of timestep evolution for VarQITE
        return int(tau / 0.01) - 1

    def split_preparation_list(self, preparation_list, index):
        r"""Splits the list of evolved statevectors results from the start to the index provided.

        Args:
            preparation_list: The list of statevector evolution results you want to split.
            index: Max index you want to split the list to

        Returns:
            Splitted list of evolved statevectors results.
        """
        final_preparation_list = copy.deepcopy(preparation_list)
        for basis_state in preparation_list.keys():
            for key in preparation_list[basis_state].keys():
                final_preparation_list[basis_state][key] = preparation_list[
                    basis_state
                ][key][: index + 1]
        return final_preparation_list

    def evolving(self, initial_state: str, tau: float):
        r"""Evolves the initial state of imaginary time tau.

        Performs the QITE algorithm to evolve the initial state to the imaginary time tau.
        It's based on the VarQITE provided by Qiskit and tailored to QMETTS needs.

        Args:
            initial_state: Label of the initial state you want to evolve.
            tau: Imaginary time you want to evolve the initial state to.

        Returns:
            Evolved statevector results, containing lists for all the imaginary times, the circuits and the parameters.
        """
        temporary_problem = TimeEvolutionProblem(self.H.get_pauli(), tau)
        temporary_qite = VarQITE(
            self.ansatz.build(),
            lb.state_to_par(
                label=initial_state, num_params=self.ansatz.get_num_parameters(),
            ),
            num_timesteps=self.num_timesteps,
        )
        temporary_result = temporary_qite.evolve(temporary_problem)
        temporary_ansatz = two_local(
            self.ansatz.num_qubits,
            self.ansatz.rotation_blocks,
            self.ansatz.entanglement_blocks,
            self.ansatz.entanglement,
            self.ansatz.num_reps,
            self.ansatz.par_name,
        )
        circuits = []
        for parameters in temporary_result.parameter_values:
            temporary_ansatz.bind_parameters(parameters)
            circuits.append(temporary_ansatz.build())
        qmetts_result = {
            "circuit_list": circuits,
            "parameter_list": temporary_result.parameter_values,
            "time_list": temporary_result.times,
        }
        return qmetts_result

    def compute_evo_on_basis(self):
        r"""Computes the evolution of all the statevectors provided of imaginary time tau.

        Args:
            basis_list: List of all the basis statevectors labels you want to evolve.
            tau: Imaginary time you want to evolve the statevectors to.

        Returns:
            Dictionary of evolved statevector results, with basis statevectors labels as keys.
        """
        preparation_result = {}
        for basis_state in self.basis_list:
            print("evolving {} basis state".format(basis_state))
            preparation_result[basis_state] = self.evolving(
                initial_state=basis_state, tau=self.beta / 2
            )
            print("done")
        with open(self.file_name, "wb") as f:
            # Pickle the preparation_result dictionary using the highest protocol available.
            pickle.dump(preparation_result, f, pickle.HIGHEST_PROTOCOL)

    def compute_exp_on_basis(self, op, preparation_result):
        r"""Computes the expectation value of the observable op on the evolved statevectors.

        Args:
            op: Observable you want to compute the expectation value as SparsePauliOp.
            preparation_result: Dictionary of the basis statevectors results.

        Returns:
            Dictionary of the expectation values with basis statevectors labels as keys.
        """
        preparation_exp_values = {}
        for basis_state in self.basis_list:
            preparation_exp_values[basis_state] = lb.exp_value(
                circuit=preparation_result[basis_state]["circuit_list"][-1],
                observable=op,
            )
        return preparation_exp_values

    def qmetts(
        self, preparation_result, preparation_exp_values, initial_state, beta, shots,
    ):
        r"""Performs the actual QMETTS algorithm.

        Starting from the initial state, builds the shots-long chain of states.
        The choice is made by measuring a random product operators on the evolved basis statevectors
        Once the chain is built, It averages the expectation values.

        Args:
            preparation_result: Dictionary containing the basis statevectors evolvution results up to beta/2.
            preparation_exp_values: Dictionary containing the expectation values on the basis statevectors evolved up to beta/2.
            initial_state: Label of the initial state you want to start the chain with.
            beta: Inverse time of the thermal average.
            shots: Length of the chain, i.e. number of times the measure is made.

        Returns:
            Dictionary containing beta, the chain, the expectation values of the chain and the thermal average.
        """
        # Choose the states
        state_list = [initial_state]
        for shot_i in range(shots):
            state_list.append(
                lb.choose_state(
                    circuit=preparation_result[state_list[-1]]["circuit_list"][-1],
                    basis_measure_list=self.basis_measure_list,
                )
            )
        # Take the exp_values
        exp_value_list = []
        for state in state_list:
            exp_value_list.append(preparation_exp_values[state])
        th_average = np.average(exp_value_list)
        result = {
            "beta": beta,
            "state_list": state_list,
            "exp_value_list": exp_value_list,
            "th_average": th_average,
        }

        return result

    def multi_beta_qmetts(self, op, initial_state, shots):
        r"""Performs the QMETTS algorithm for all the betas.

        For each intermediate beta, It splits the results of the evolved basis statevectors and performs the QMETTS algorithm.

        Args:
            op: Observable you want to make the thermal average of, as SparsePauliOp.
            initial_state: Label of the initial state you want to start each chain with.
            shots: Length of each chain, i.e. number of times the measure is made.

        Returns:
            Results of the QMETTS algorithm for all the betas, as QMETTS_result class.
        """
        # tau = self.beta / 2
        print("Data reading started")
        with open(self.file_name, "rb") as f:
            preparation_result = pickle.load(f)
        print("Data reading finished")
        # preparation_result = self.compute_evo_on_basis(
        #     tau=tau, basis_list=self.basis_list
        # )
        preparation_exp_values = {}
        for basis_state in self.basis_list:
            preparation_exp_values[basis_state] = []
        multi_beta_qmetts_result = []
        tau_list = self.get_tau_list()
        for partial_tau in tau_list:
            temporary_preparation_result = self.split_preparation_list(
                preparation_list=preparation_result,
                index=self.return_tau_index(tau=partial_tau),
            )
            print("computing exp_value for beta = {}".format(partial_tau * 2))
            temporary_preparation_exp_values = self.compute_exp_on_basis(
                op=op, preparation_result=temporary_preparation_result
            )
            for basis_state in self.basis_list:
                preparation_exp_values[basis_state] += [
                    temporary_preparation_exp_values[basis_state]
                ]
            multi_beta_qmetts_result.append(
                self.qmetts(
                    preparation_result=temporary_preparation_result,
                    preparation_exp_values=temporary_preparation_exp_values,
                    initial_state=initial_state,
                    beta=partial_tau * 2,
                    shots=shots,
                )
            )
        QMETTS_result = result_handler.QMETTS_results(
            N=self.N,
            gy=self.gy,
            B=self.B,
            preparation_result=preparation_result,
            preparation_exp_values=preparation_exp_values,
            multi_beta_qmetts_result=multi_beta_qmetts_result,
        )

        return QMETTS_result
