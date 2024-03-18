# -*- coding: utf-8 -*-
r"""
Script for collecting data about my thesis on thermal states generation and analysis of some TD properties for the LMG model.



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


for N in [2]:
    for gy in [0.0]:
        for B in [0.1]:
            print("Simulation for N = {}, gy = {}, B = {}".format(N, gy, B))
            # N = 3
            # gy = 0.5
            # B = 0.1
            first_beta = 0.2
            last_beta = 5.0
            num_beta_points = 4
            betas = np.linspace(first_beta, last_beta, num_beta_points)
            maxiter = 110
            optimizer = "COBYLA"
            H = LMG_hamiltonian(N, gy, B)
            ancilla_ansatz = two_local(num_qubits=N, num_reps=1, entanglement="linear")
            system_ansatz = two_local(
                num_qubits=N, num_reps=1, entanglement="linear", par_name="y"
            )

            flag = "statevector"
            model_tag = None
            if flag == "noise":
                model_tag = FakeManila()
            elif flag == "hardware":
                model_tag = "ibm_osaka"
            backend = setup.setup_backend(flag=flag, model_tag=model_tag)
            path = "./MHETS_data/"
            file_name = setup.setup_file_name(H=H, flag=flag)
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
            }

            mhets = MHETS_instance(
                H=H,
                ancilla_ansatz=ancilla_ansatz,
                system_ansatz=system_ansatz,
                optimization_options=optimization_options,
                flag=flag,
                backend=backend,
            )

            try:
                with open(path + file_name, "rb") as f:
                    old_data = pickle.load(f)
            except:
                print("No file found. Optimize from scratch")
                multi_beta_result = mhets.multi_beta_optimization_from_scratch(betas)
            else:
                print("File found. Append results")
                new_betas = setup.setup_betas(old_betas=old_data["betas"], betas=betas)
                print("old_betas", old_data["betas"])
                print("betas_inserted", betas)
                print("final_beta_list", new_betas)
                # INITIALIZE DICTIONARY RESULT
                multi_beta_result = {
                    "optimization_options": mhets.optimization_options,
                    "backend": mhets.backend,
                }
                for key in old_data.keys():
                    if key != "optimization_options":
                        if key != "backend":
                            multi_beta_result[key] = []
                # FILLING DICTIONARY RESULT
                for beta in new_betas:
                    if beta in old_data["betas"]:
                        print("beta beta check here, continue implementation")
                    else:
                        # TODO: check initial parameters when flag=hardware, not implemented
                        result = mhets.optimize(
                            beta=beta,
                            maxiter=mhets.optimization_options["maxiter"],
                            optimizer=mhets.optimization_options["optimizer"],
                        )
                        multi_beta_result["betas"].append(beta)
                        for key in result.keys():
                            multi_beta_result[key].append(result[key])
            # WRITING DATA TO FILE
            path = "./MHETS_data/"
            file_name = setup.setup_file_name(H, flag)
            with open(path + file_name, "wb") as f:
                # Pickle the preparation_result dictionary using the highest protocol available.
                pickle.dump(multi_beta_result, f, pickle.HIGHEST_PROTOCOL)
# def fill():
#     control if there is file
#     if there is:
#         old_multi_beta_result
#         new long beta list
#         for beta in new betas:
#             if beta in old_result[betas]:
#                 partial_result = take_partial_results_from_old_result(beta)
#             else:
#                 partial_result = optimize
#             fill multi_beta_result
#     else:
#         all standard
