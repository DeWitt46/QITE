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
from library.ansatz_creation import two_local, pma


flag = "noise"
for N in [3]:
    for gy in [0.49]:
        for B in [0.2]:
            for shots in [100]:  # Ignore If flag == statevector
                print("----------------------------------------")
                print("Simulation for N = {}, gy = {}, B = {}".format(N, gy, B))
                print("----------------------------------------")
                first_beta = 0.2
                last_beta = 5.0
                num_beta_points = 3
                betas = np.linspace(first_beta, last_beta, num_beta_points)
                maxiter = 25
                optimizer = "spsa"
                tol = 1e-4
                H = LMG_hamiltonian(N, gy, B)
                ancilla_ansatz = two_local(
                    num_qubits=N, num_reps=1, entanglement="linear"
                )
                system_ansatz = pma(
                    num_qubits=N, num_reps=0, architecture="linear", par_name="y"
                )

                run_flag = True  # True if you want some randomized starting point for minimization
                n_starting_point = 3  # Ignore if run_flag is False
                model_tag = None

                if flag == "noise":
                    model_tag = FakeManila()
                elif flag == "hardware":
                    model_tag = "ibm_osaka"
                backend = setup.setup_backend(flag=flag, model_tag=model_tag)
                path = "./MHETS_data/"
                initial_parameter_list = setup.setup_initial_parameter_list(
                    H=H, flag=flag, num_beta_points=len(betas), path=path
                )
                optimization_options = setup.setup_optimization_options(
                    ancilla_ansatz=ancilla_ansatz,
                    system_ansatz=system_ansatz,
                    maxiter=maxiter,
                    optimizer=optimizer,
                    initial_parameter_list=initial_parameter_list,
                    flag=flag,
                    tol=tol,
                    shots=shots,
                )
                pma_flag = False
                if optimization_options["system_ansatz"] == "pma":
                    pma_flag = True
                file_name = setup.setup_file_name(
                    H=H, flag=flag, shots=shots, pma_flag=pma_flag
                )
                if flag == "noise":
                    file_name = file_name.replace(
                        ".pickle", "_{}.pickle".format(optimizer)
                    )
                mhets = MHETS_instance(
                    H=H,
                    ancilla_ansatz=ancilla_ansatz,
                    system_ansatz=system_ansatz,
                    optimization_options=optimization_options,
                    flag=flag,
                    backend=backend,
                )

                if run_flag is True:
                    (
                        multi_beta_result,
                        multi_beta_runs,
                    ) = mhets.multi_beta_optimization_run(
                        betas=betas, n_starting_point=n_starting_point
                    )
                else:
                    try:
                        with open(path + file_name, "rb") as f:
                            old_data = pickle.load(f)
                    except FileNotFoundError:
                        print("No file found. Optimize from scratch")
                        multi_beta_result = mhets.multi_beta_optimization_from_scratch(
                            betas
                        )
                    else:
                        print("File found. Append results")
                        new_betas = setup.setup_betas(
                            old_betas=old_data["betas"], betas=betas
                        )
                        print("old_betas", old_data["betas"])
                        print("betas_inserted", betas)
                        print("final_beta_list", new_betas)
                        multi_beta_result = mhets.multi_beta_optimization_from_data(
                            betas=new_betas, old_data=old_data
                        )
                # WRITING DATA TO FILE
                path = "./MHETS_data/"
                # IF YOU WANT TO CHANGE SOMETHING IN THE FILE NAME
                # file_name = setup.setup_file_name(H, flag, shots=shots, pma_flag=pma_flag)
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
                        if pma_flag == True:
                            file_name.replace(".pickle", "_pma.pickle")
                        with open(path + file_name, "wb") as f:
                            # Pickle the preparation_result dictionary using the highest protocol available.
                            pickle.dump(
                                multi_beta_runs[run_index], f, pickle.HIGHEST_PROTOCOL
                            )
