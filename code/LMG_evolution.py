# -*- coding: utf-8 -*-
r"""
Script that contains some tries on LMG model's time evolution.



Created on Sat Dec  9 17:33:31 2023

@author: DeWitt
"""
import numpy as np

from qiskit.quantum_info import Statevector


from time_evolution_problem import TimeEvolutionProblem
from QITE.var_qite import VarQITE
from library.ansatz_creation import two_local
from library.operator_creation import LMG_hamiltonian


N = 2
B = 0.7
gy = 0.1
beta = 2.0

twolocal = two_local(N)
ansatz = twolocal.build()
H = LMG_hamiltonian(N=N, gy=gy, B=B)
parameter_list = np.zeros(len(twolocal.get_parameters()))
for i in range(len(parameter_list)):
    parameter_list[i] = 2 * np.pi * np.random.uniform()
problem = TimeEvolutionProblem(H.get_pauli(), beta)
qite = VarQITE(ansatz, parameter_list, num_timesteps=None)
result = qite.evolve(problem)


print(result)
twolocal.bind_parameters(result.parameter_values[0])
initial_state = Statevector([1.0, 0.0, 0.0, 0.0]).evolve(twolocal.build())
final_state = Statevector([1.0, 0.0, 0.0, 0.0]).evolve(result.evolved_state)
exact_ground_state = H.get_ground_state()[-1]
print(initial_state)
print(final_state)
print(exact_ground_state)

print(type(H))
