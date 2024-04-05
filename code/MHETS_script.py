# -*- coding: utf-8 -*-
r"""
Script for collecting data about my thesis on thermal states generation and analysis of 
some TD properties for the LMG model.



Created on Thu Feb 29 00:06:42 2024

@author: DeWitt
"""
import pickle
import numpy as np


from qiskit.providers.fake_provider import FakeManila


from library import setup
from library.MHETS import MHETS_instance
from library.operator_creation import LMG_hamiltonian
from library.ansatz_creation import two_local


for N in [4]:
    for gy in [0.0]:
        for B in [0.2]:
            for shots in [100]:
                print("Simulation for N = {}, gy = {}, B = {}".format(N, gy, B))
                # N = 3
                # gy = 0.5
                # B = 0.1
                first_beta = 0.2
                last_beta = 5.0
                num_beta_points = 1
                betas = np.linspace(first_beta, last_beta, num_beta_points)
                maxiter = 30
                optimizer = "COBYLA"
                tol = 1.5e-1
                # shots = 1000  # Ignore If flag == statevector
                H = LMG_hamiltonian(N, gy, B)
                ancilla_ansatz = two_local(num_qubits=N, num_reps=1, entanglement="linear")
                system_ansatz = two_local(
                    num_qubits=N, num_reps=2, entanglement="linear", par_name="y"
                )

                flag = "qasm"
                run_flag = (
                    False  # True if you want some randomized starting point for minimization
                )
                n_starting_point = 10  # Ignore if run_flag is False
                model_tag = None

                if flag == "noise":
                    model_tag = FakeManila()
                elif flag == "hardware":
                    model_tag = "ibm_osaka"
                backend = setup.setup_backend(flag=flag, model_tag=model_tag)
                path = "./MHETS_data/"
                file_name = setup.setup_file_name(H=H, flag=flag, shots=shots)
                initial_parameter_list = setup.setup_initial_parameter_list(
                    H=H, flag=flag, num_beta_points=len(betas), path=path
                )

                optimization_options = {
                    "ancilla_ansatz": "two_local",
                    "ancilla_num_reps": ancilla_ansatz.get_num_reps(),
                    "ancilla_ansatz_entanglement": ancilla_ansatz.get_entanglement(),
                    "ancilla_ansatz_rotation_blocks": ancilla_ansatz.get_rotation_blocks(),
                    "ancilla_ansatz_entanglement_blocks": ancilla_ansatz.get_entanglement_blocks(),
                    "system_ansatz": "two_local",
                    "system_num_reps": system_ansatz.get_num_reps(),
                    "system_ansatz_entanglement": system_ansatz.get_entanglement(),
                    "system_ansatz_entanglement_blocks": system_ansatz.get_entanglement_blocks(),
                    "system_ansatz_rotation_blocks": system_ansatz.get_rotation_blocks(),
                    "maxiter": maxiter,
                    "optimizer": optimizer,
                    "initial_parameter_list": initial_parameter_list,
                    "flag": flag,
                    "tol": tol,
                    "shots": shots,
                }

                mhets = MHETS_instance(
                    H=H,
                    ancilla_ansatz=ancilla_ansatz,
                    system_ansatz=system_ansatz,
                    optimization_options=optimization_options,
                    flag=flag,
                    backend=backend,
                )

                if run_flag is True:
                    multi_beta_result, multi_beta_runs = mhets.multi_beta_optimization_run(
                        betas=betas, n_starting_point=n_starting_point
                    )
                else:
                    try:
                        with open(path + file_name, "rb") as f:
                            old_data = pickle.load(f)
                    except FileNotFoundError:
                        print("No file found. Optimize from scratch")
                        multi_beta_result = mhets.multi_beta_optimization_from_scratch(betas)
                    else:
                        print("File found. Append results")
                        new_betas = setup.setup_betas(old_betas=old_data["betas"], betas=betas)
                        print("old_betas", old_data["betas"])
                        print("betas_inserted", betas)
                        print("final_beta_list", new_betas)
                        multi_beta_result = mhets.multi_beta_optimization_from_data(
                            betas=new_betas, old_data=old_data
                        )
                # WRITING DATA TO FILE
                path = "./MHETS_data/"
                file_name = setup.setup_file_name(H, flag, shots=shots)
                with open(path + file_name, "wb") as f:
                    # Pickle the preparation_result dictionary using the highest protocol available.
                    pickle.dump(multi_beta_result, f, pickle.HIGHEST_PROTOCOL)
                if run_flag is True:
                    for run_index in range(len(multi_beta_runs)):
                        path = "./MHETS_data/"
                        file_name = (
                            "MHETS_{}at_gy{}_B{}_run{}".format(
                                H.N, H.gy, H.B, run_index
                            ).replace(".", "")
                        ) + ".pickle"
                        with open(path + file_name, "wb") as f:
                            # Pickle the preparation_result dictionary using the highest protocol available.
                            pickle.dump(multi_beta_runs[run_index], f, pickle.HIGHEST_PROTOCOL)
