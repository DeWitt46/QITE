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


N = 4
gy = 0.5
B = 0.1
final_beta = 3.0
num_beta_points = 10
shots = 5000
initial_state = "+00+"
H = LMG_hamiltonian(N, gy, B)
operators = ["zxxz", "xzzx"]
flag = "manual"
ansatz = two_local(num_qubits=N, num_reps=3)
qmetts_instance = QMETTS_instance(
    H, operators, flag, final_beta, num_beta_points, shots, initial_state, ansatz,
)
print(qmetts_instance.get_basis_list())
print(qmetts_instance.get_basis_measure_list())
# qmetts_instance.compute_evo_on_basis()
QMETTS_result = qmetts_instance.multi_beta_qmetts(
    op=H.get_pauli(), initial_state=initial_state, shots=shots
)
plotting.plot_thermal_average(QMETTS_result)
plotting.plot_state_histogram(QMETTS_result)
# plotting.plot_qite(QMETTS_result, qmetts_instance.get_basis_list())


# 1st  all
# 2nd  xxzz, zzxx
# 3rd  xzzx, zxxz
# 4th  zzxx, xxzz
# 5th  xzxz, zxzx
# 6th  xzzz, zxxx
# 7th  zxzz, xzxx
# 8th  zzxz, xxzx
# 9th  zzzx, xxxz
# 10th 7+3
# 11th zxzz, xzxx, xzzx
# 12th xzzx, zxxz, zxzz
# 13th xzzx, zxxz, zxzz, xzzz, zxxx
