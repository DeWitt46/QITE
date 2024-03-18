# -*- coding: utf-8 -*-
r"""
Script to add/change something in the data collected



Created on Fri Mar  1 11:03:55 2024

@author: DeWitt
"""
import pickle


for N in [2, 3]:
    for gy in [0.1, 0.6]:
        for B in [0.15]:
            optimization_options = {
                "ancilla_ansatz": "two_local",
                "ancilla_num_reps": 2,
                "ancilla_ansatz_entanglement": "linear",
                "ancilla_ansatz_entanglement_blocks": ["cx"],
                "ancilla_ansatz_rotation_blocks": ["ry"],
                "system_ansatz": "two_local",
                "system_num_reps": 2,
                "system_ansatz_entanglement": "linear",
                "system_ansatz_entanglement_blocks": ["cx"],
                "system_ansatz_rotation_blocks": ["ry"],
                "maxiter": 5000,
                "optimizer": "COBYLA",
            }
            path = "./MHETS_data/"
            file_name = (
                "MHETS_{}at_gy{}_B{}".format(N, gy, B).replace(".", "") + ".pickle"
            )
            with open(path + file_name, "rb") as f:
                multi_beta_result = pickle.load(f)
            multi_beta_result["optimization_options"] = optimization_options
            with open(path + file_name, "wb") as f:
                pickle.dump(multi_beta_result, f, pickle.HIGHEST_PROTOCOL)
