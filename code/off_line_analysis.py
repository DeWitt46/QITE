# -*- coding: utf-8 -*-
r"""
Script to handle off-line analysis of MHETS algorithm


Created on Thu Feb 29 00:28:15 2024

@author: DeWitt
"""
import pickle
import numpy as np


from qiskit.quantum_info import partial_trace
from library import plotting, setup
from library.operator_creation import LMG_hamiltonian
from library.MHETS import MHETS_instance

flag = "statevector"
N = 4
gy = 0.0
B = 0.2
H = LMG_hamiltonian(N, gy, B)


path = "./MHETS_data/"
file_name = setup.setup_file_name(H, flag, shots=10)
# IF run_flag WAS TRUE
# run_index = 3
# file_name = (
#     "MHETS_{}at_gy{}_B{}_run{}".format(H.N, H.gy, H.B, run_index).replace(".", "")
# ) + ".pickle"


# IF PLOTTING SINGLE MULTI_BETA_RESULT
with open(path + file_name, "rb") as f:
    multi_beta_result = pickle.load(f)

optimization_options = multi_beta_result["optimization_options"]
try:
    a = optimization_options["shots"]
except KeyError:
    optimization_options["shots"] = 1024
beta_list = multi_beta_result["betas"]


ancilla_ansatz, system_ansatz = setup.setup_ansatz(N, optimization_options)
mhets = MHETS_instance(
    H, ancilla_ansatz, system_ansatz, optimization_options=optimization_options
)
rho_s_list = []
for index in range(len(beta_list)):
    mhets.update_parameters(multi_beta_result["optimized_parameter_list"][index])
    rho = mhets.QST(mhets.build_total_circuit(), shots=optimization_options["shots"])
    rho_s_list.append(partial_trace(rho, range(ancilla_ansatz.get_num_qubits())))
# backend variable is just for the name that has to be in the plots, no calculations
plotting.plot_MHETS_thermal_average(
    beta_list, rho_s_list, H, backend="FakeManila", beta_final=10
)
plotting.plot_fidelity(beta_list, rho_s_list, H, backend="FakeManila")

for index in range(len(beta_list)):
    print("beta*Helmoltz energy expected vs simulated")
    print(H.cost_function(beta_list[index]), multi_beta_result["Helmoltz energy"][index])
    print("")
# print(multi_beta_result.keys())
print(multi_beta_result["optimized_parameter_list"])
# FOR NOISE SIMULATIONS
# for data in multi_beta_result["callback_data"][0]:
# data = multi_beta_result["callback_data"][0][0]
# print(data.analysis_results("state_fidelity"))


# IF PLOTTING MULTI_DATA
# file_names = []
# flag = "qasm"
# path = "./MHETS_data/"
# for N in [2]:
#     for gy in [0.0]:
#         for B in [0.2]:
#             H = LMG_hamiltonian(N, gy, B)
#             for shots in [10, 100, 1000, 10000]:
#                 file_names.append(setup.setup_file_name(H, flag, shots))

# multi_data = setup.setup_offline_multi_data(path, file_names)
# plotting.plot_MHETS_shots(multi_data=multi_data, betas=np.linspace(0.2, 5.0, 3), H=H)
# plotting.plot_MHETS_shots_fidelity(multi_data=multi_data, betas=np.linspace(0.2, 5.0, 3), H=H)
