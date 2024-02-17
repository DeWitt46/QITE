# -*- coding: utf-8 -*-
r"""
Script for collecting data about my thesis on thermal states generation and analysis of some TD properties for the LMG model.



Created on Wed Jan 10 18:44:16 2024

@author: DeWitt
"""
import numpy as np
import matplotlib.pyplot as plt


# from library import state_label as lb
# from library import result_handler
from library import plotting
from library.ansatz_creation import two_local
from library.operator_creation import LMG_hamiltonian
from library.QMETTS import QMETTS_instance


N = 5
gy = 0.5
B = 0.8
final_beta = 3.0
num_beta_points = 10
shots = 1024
initial_state = "00000"
H = LMG_hamiltonian(N, gy, B)
operators = ["zzzzz"]
flag = "manual"
ansatz = two_local(num_qubits=N, num_reps=2)
qmetts_instance = QMETTS_instance(
    H,
    operators,
    flag,
    final_beta,
    num_beta_points,
    shots,
    initial_state,
    ansatz,
)
print(qmetts_instance.get_basis_list())
print(qmetts_instance.get_basis_measure_list())
QMETTS_result = qmetts_instance.multi_beta_qmetts(
    op=H.get_pauli(), initial_state=initial_state, shots=shots
)
plotting.plot_thermal_average(QMETTS_result)
plotting.plot_state_histogram(QMETTS_result)
plotting.plot_qite(QMETTS_result)
