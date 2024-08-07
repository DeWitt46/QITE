# -*- coding: utf-8 -*-
r"""
Class to generate an LMG Hamiltonian and numerically get eigenstates, thermal averages, ecc.



Created on Sat Dec  9 17:40:05 2023

@author: DeWitt
"""

# TODO: Add error raising
import numpy as np
import time
from qiskit.quantum_info import SparsePauliOp


class LMG_hamiltonian:
    def __init__(self, N: int, gy: float, B: float):
        self.N = N
        self.gy = gy
        self.B = B
        self.pauli_list, self.coeff_list = self.op_list(N=self.N, gy=self.gy, B=self.B)
        self.pauli = SparsePauliOp(self.pauli_list, self.coeff_list)

    def get_pauli(self):
        return self.pauli

    def get_matrix(self):
        return self.pauli.to_matrix()

    def get_ground_state(self):
        eigenvalues, eigenstates = self.diagonalize(self.pauli.to_matrix())
        return eigenvalues[0], eigenstates[0]

    def get_eigenstates(self):
        eigenvalues, eigenstates = self.diagonalize(self.pauli.to_matrix())
        return eigenvalues, eigenstates

    def get_partition_function(self, beta):
        Z, rho = self.thermalize(self.get_matrix(), beta)
        return Z

    def get_thermal_state(self, beta):
        Z, rho = self.thermalize(self.get_matrix(), beta)
        return rho

    def op_list(self, N: int, gy: float, B: float):
        field_list = []
        x_interaction_list = []
        y_interaction_list = []
        for spin_i in range(N):
            if B != 0.00:
                field_list.append(spin_i * "I" + "Z" + (N - spin_i - 1) * "I")
            for spin_j in range(N):
                if spin_j > spin_i:
                    x_interaction_list.append(
                        spin_i * "I"
                        + "X"
                        + (spin_j - spin_i - 1) * "I"
                        + "X"
                        + (N - spin_j - 1) * "I"
                    )
                    if gy != 0.00:
                        y_interaction_list.append(
                            spin_i * "I"
                            + "Y"
                            + (spin_j - spin_i - 1) * "I"
                            + "Y"
                            + (N - spin_j - 1) * "I"
                        )
        coeff_list = (
            [-B] * len(field_list)
            + [-1.0 / N] * len(x_interaction_list)
            + [-gy / N] * len(y_interaction_list)
        )
        return field_list + x_interaction_list + y_interaction_list, coeff_list

    def diagonalize(self, op):
        start = time.time()
        w, v = np.linalg.eigh(op)
        eigenvalues = []
        eigenstates = []
        for i in range(len(w)):  # Just sorting eigenthings
            eigenvalues.append(w[i])
            eigenstates.append(v[:, i])
        self.time_to_diagonalize = time.time() - start
        return eigenvalues, eigenstates

    def thermalize(self, H, beta):
        eigenvalues, eigenstates = self.diagonalize(H)
        Z = 0.0
        rho = np.zeros([np.shape(H)[0], np.shape(H)[0]], dtype=complex)
        for i in range(len(eigenvalues)):
            Z += np.exp(-beta * eigenvalues[i])
            rho += np.exp(-beta * eigenvalues[i]) * np.matmul(
                np.atleast_2d(eigenstates[i]).T, np.atleast_2d(eigenstates[i])
            )
        rho = rho / Z
        return Z, rho

    def thermal_average(self, op, beta):
        average = np.trace(op @ self.get_thermal_state(beta))
        return average

    def cost_function(self, beta):
        entropy = 0.0
        eigenvalues, eigenstates = self.diagonalize(self.get_thermal_state(beta))
        for eigenvalue in eigenvalues:
            if eigenvalue != 0.0:
                entropy -= eigenvalue * np.log(eigenvalue)
        energy = self.thermal_average(self.get_matrix(), beta)
        return np.real(beta * energy - entropy)

    def get_sqrt(self, op):  # Not working
        eigenvalues, eigenstates = self.diagonalize(op)
        op_sqrt = np.zeros(op.shape, dtype=np.complex128)
        for index in range(len(eigenvalues)):
            op_sqrt += np.sqrt(eigenvalues[index]) * (
                eigenstates[index] @ np.matrix.getH(eigenstates[index])
            )
        return op_sqrt

    def relative_entropy(self, first, second):
        """
            S(first||second) = sum_i p_first_i log(p_first_i/p_second_i)
        """
        S = 0.0

        for i in range(2 ** self.N):
            p_first_i = first[i][i]
            p_second_i = second[i][i]
            if p_second_i != 0.0:
                if p_first_i != 0:
                    S += p_first_i * (np.log(p_first_i) - np.log(p_second_i))
        return S
