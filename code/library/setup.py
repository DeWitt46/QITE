# -*- coding: utf-8 -*-
r"""
Collection of functions to handle the preparation of the MHETS instance 
to make the main script more readable



Created on Tue Mar  5 18:44:47 2024

@author: DeWitt
"""
import pickle


from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator


from library.ansatz_creation import two_local


def setup_ansatz(N, optimization_options):
    if optimization_options["ancilla_ansatz"] == "two_local":
        ancilla_ansatz = two_local(
            num_qubits=N,
            rotation_blocks=optimization_options["ancilla_ansatz_rotation_blocks"],
            entanglement_blocks=optimization_options[
                "ancilla_ansatz_entanglement_blocks"
            ],
            entanglement=optimization_options["ancilla_ansatz_entanglement"],
            num_reps=optimization_options["ancilla_num_reps"],
        )
    if optimization_options["system_ansatz"] == "two_local":
        system_ansatz = two_local(
            num_qubits=N,
            rotation_blocks=optimization_options["system_ansatz_rotation_blocks"],
            entanglement_blocks=optimization_options[
                "system_ansatz_entanglement_blocks"
            ],
            entanglement=optimization_options["system_ansatz_entanglement"],
            num_reps=optimization_options["system_num_reps"],
            par_name="y",
        )
    return ancilla_ansatz, system_ansatz


def setup_backend(flag="statevector", model_tag=None):
    if flag == "statevector":
        backend = None
    elif flag == "noise":
        # Make a noise model
        fake_backend = model_tag
        noise_model = NoiseModel.from_backend(fake_backend)
        backend = AerSimulator(method="density_matrix", noise_model=noise_model)
    elif flag == "hardware":
        provider = IBMProvider()
        backend = provider.get_backend(model_tag)  # Choose your backend
    return backend


def setup_initial_parameter_list(
    H, flag="statevector", num_beta_points=4, path="./MHETS_data/"
):
    if flag == "hardware":
        # Takes optimized parameters from statevector simulation
        file_name = setup_file_name(H, flag="statevector")
        with open(path + file_name, "rb") as f:
            multi_beta_result_sv = pickle.load(f)
        initial_parameter_list = multi_beta_result_sv["optimized_parameter_list"]
    else:
        initial_parameter_list = [None for i in range(num_beta_points)]
    return initial_parameter_list


def setup_betas(old_betas, betas):
    new_betas = old_betas.copy()
    for beta in betas:
        try:
            index = new_betas.index(beta)
        except:
            new_betas.append(beta)
        else:
            if index < (len(new_betas) - 1):
                new_betas.append((beta + new_betas[index + 1]) / 2.0)
            else:
                new_betas.append(beta + (beta - new_betas[index - 1]) / 2.0)
        new_betas.sort()
    return new_betas


def setup_file_name(H, flag="statevector"):
    if flag == "statevector":
        file_name = (
            "MHETS_{}at_gy{}_B{}".format(H.N, H.gy, H.B).replace(".", "")
        ) + ".pickle"
    elif flag == "noise":
        file_name = (
            "MHETS_{}at_gy{}_B{}_noise".format(H.N, H.gy, H.B).replace(".", "")
        ) + ".pickle"
    elif flag == "hardware":
        file_name = (
            "MHETS_{}at_gy{}_B{}_hardware".format(H.N, H.gy, H.B).replace(".", "")
        ) + ".pickle"
    return file_name
