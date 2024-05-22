# -*- coding: utf-8 -*-
r"""
Script for collecting data about my thesis on thermal states generation and analysis of 
some TD properties for the LMG model.



Created on Thu Feb 29 00:06:42 2024

@author: DeWitt
"""
import os
import pickle
import numpy as np


from qiskit.providers.fake_provider import FakeManila


from library import setup
from library.MHETS import MHETS_instance
from library.operator_creation import LMG_hamiltonian
from library.ansatz_creation import two_local, pma

# __main__
# path
# file_name
# if path+file_name exists:
#     change betas
#     multi_beta_sim
# elif path+file_name does not exist
#     multi_beta_sim

# __multi_beta_sim___
# check_temp_dir
# parallel single_beta_sim
# build_multi_beta_result
# write_data_to_file(multi_beta_result)
# delete temp_dir

# __build_multi_beta_result__
# multi_beta_result = first_single_beta_result.copy()
# for other_single_beta_file in temp_dir
#     take single_beta_result
#     append each list to multi_beta_result


def write_data_to_file(
    path, file_name, multi_beta_result, run_flag=False, pma_flag=False, multi_beta_runs=None
):
    def check_path(path):
        if not os.path.isdir(path):
            os.mkdir(path)
            print("Built directory: {}".format(path))

    check_path(path)
    with open(path + file_name, "wb") as f:
        # Pickle the preparation_result dictionary using the highest protocol available.
        pickle.dump(multi_beta_result, f, pickle.HIGHEST_PROTOCOL)
    if run_flag is True:
        for run_index in range(len(multi_beta_runs)):
            path = "./MHETS_data/"
            file_name = (
                "MHETS_{}at_gy{}_B{}_run{}".format(H.N, H.gy, H.B, run_index).replace(".", "")
            ) + ".pickle"
            if pma_flag is True:
                file_name.replace(".pickle", "_pma.pickle")
            with open(path + file_name, "wb") as f:
                # Pickle the preparation_result dictionary using the highest protocol available.
                pickle.dump(multi_beta_runs[run_index], f, pickle.HIGHEST_PROTOCOL)


flag = "statevector"
for N in [2]:
    for gy in [0.36]:
        for B in [0.1]:
            for shots in [100]:  # Ignore If flag == statevector
                print("----------------------------------------")
                print("Simulation for N = {}, gy = {}, B = {}".format(N, gy, B))
                print("----------------------------------------")
                first_beta = 0.2
                last_beta = 5.0
                num_beta_points = 5
                betas = np.linspace(first_beta, last_beta, num_beta_points)
                maxiter = 10000
                optimizer = "cobyla"
                tol = 1e-6
                H = LMG_hamiltonian(N, gy, B)
                ancilla_ansatz = two_local(num_qubits=N, num_reps=1, entanglement="linear")
                system_ansatz = pma(
                    num_qubits=N, num_reps=1, architecture="linear", par_name="y"
                )

                run_flag = (
                    False  # True if you want some randomized starting point for minimization
                )
                n_starting_point = 3  # Ignore if run_flag is False
                model_tag = None

                if flag == "noise":
                    model_tag = FakeManila()
                elif flag == "hardware":
                    model_tag = "ibm_osaka"
                backend = setup.setup_backend(flag=flag, model_tag=model_tag)
                path = "./MHETS_data/"
                # CHOOSE FILE_NAME
                pma_flag = False
                if system_ansatz.get_name() == "pma":
                    pma_flag = True
                file_name = setup.setup_file_name(H=H, flag=flag, shots=shots, pma_flag=pma_flag)
                if flag == "noise":
                    file_name = file_name.replace(".pickle", "_{}.pickle".format(optimizer))
                # SETUP INITIAL PARAMETER LIST
                initial_parameter_list = setup.setup_initial_parameter_list(
                    H=H,
                    flag=flag,
                    pma_flag=pma_flag,
                    num_beta_points=len(betas),
                    path=path,
                    take_statevector_initial_parameters=False,
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

                mhets = MHETS_instance(
                    H=H,
                    ancilla_ansatz=ancilla_ansatz,
                    system_ansatz=system_ansatz,
                    optimization_options=optimization_options,
                    flag=flag,
                    backend=backend,
                )

                # RUNNING SIMULATION
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
                file_name = setup.setup_file_name(H, flag, shots=shots, pma_flag=pma_flag)
                try:
                    write_data_to_file(
                        path=path,
                        file_name=file_name,
                        multi_beta_result=multi_beta_result,
                        run_flag=run_flag,
                        pma_flag=pma_flag,
                        multi_beta_runs=multi_beta_runs,
                    )
                except NameError:
                    write_data_to_file(
                        path=path,
                        file_name=file_name,
                        multi_beta_result=multi_beta_result,
                        run_flag=run_flag,
                        pma_flag=pma_flag,
                    )
