# -*- coding: utf-8 -*-
r"""
Script to handle off-line analysis of MHETS algorithm


Created on Thu Feb 29 00:28:15 2024

@author: DeWitt
"""
import pickle


from qiskit.quantum_info import partial_trace
from library import plotting, setup
from library.operator_creation import LMG_hamiltonian
from library.MHETS import MHETS_instance

flag = "statevector"
N = 2
gy = 0.0
B = 0.1
H = LMG_hamiltonian(N, gy, B)


path = "./MHETS_data/"
file_name = setup.setup_file_name(H, flag)
with open(path + file_name, "rb") as f:
    multi_beta_result = pickle.load(f)
optimization_options = multi_beta_result["optimization_options"]
beta_list = multi_beta_result["betas"]


ancilla_ansatz, system_ansatz = setup.setup_ansatz(N, optimization_options)
mhets = MHETS_instance(
    H, ancilla_ansatz, system_ansatz, optimization_options=optimization_options
)
rho_s_list = []
for index in range(len(beta_list)):
    mhets.update_parameters(multi_beta_result["optimized_parameter_list"][index])
    rho = mhets.QST(mhets.build_total_circuit())
    rho_s_list.append(partial_trace(rho, range(ancilla_ansatz.get_num_qubits())))
# backend variable is just for the name that has to be in the plots, no calculations
plotting.plot_MHETS_thermal_average(beta_list, rho_s_list, H, backend="FakeManila")
plotting.plot_fidelity(beta_list, rho_s_list, H, backend="FakeManila")

# for index in range(len(beta_list)):
#     print(H.cost_function(beta_list[index]))
#     print(multi_beta_result["Helmoltz energy"][index])
print("")
print(multi_beta_result["betas"])
