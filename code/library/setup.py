# -*- coding: utf-8 -*-
r"""
Collection of functions to handle the preparation of the MHETS instance 
to make the main script more readable



Created on Tue Mar  5 18:44:47 2024

@author: DeWitt
"""
import pickle
import re


from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import partial_trace


from library.ansatz_creation import two_local, pma
from library.operator_creation import LMG_hamiltonian
from library.MHETS import MHETS_instance


def get_ansatz_options(ansatz):
    ansatz_options = {
        "ansatz": ansatz.get_name(),
        "num_reps": ansatz.get_num_reps(),
    }
    if ansatz_options["ansatz"] == "two_local":
        ansatz_options["ansatz_entanglement"] = ansatz.get_entanglement()
        ansatz_options["ansatz_rotation_blocks"] = ansatz.get_rotation_blocks()
        ansatz_options["ansatz_entanglement_blocks"] = ansatz.get_entanglement_blocks()
    elif ansatz_options["ansatz"] == "pma":
        ansatz_options["ansatz_architecture"] = ansatz.get_architecture()
    return ansatz_options


def setup_optimization_options(
    ancilla_ansatz,
    system_ansatz,
    maxiter,
    optimizer,
    initial_parameter_list,
    flag,
    tol,
    shots,
):
    optimization_options = {}
    for key in get_ansatz_options(ancilla_ansatz):
        optimization_options["ancilla_" + key] = get_ansatz_options(ancilla_ansatz)[key]
    for key in get_ansatz_options(system_ansatz):
        optimization_options["system_" + key] = get_ansatz_options(system_ansatz)[key]
    optimization_options["maxiter"] = maxiter
    optimization_options["optimizer"] = optimizer
    optimization_options["initial_parameter_list"] = initial_parameter_list
    optimization_options["flag"] = flag
    optimization_options["tol"] = tol
    optimization_options["shots"] = shots

    return optimization_options


def setup_ansatz(N, optimization_options):
    if optimization_options["ancilla_ansatz"] == "two_local":
        ancilla_ansatz = two_local(
            num_qubits=N,
            rotation_blocks=optimization_options["ancilla_ansatz_rotation_blocks"],
            entanglement_blocks=optimization_options["ancilla_ansatz_entanglement_blocks"],
            entanglement=optimization_options["ancilla_ansatz_entanglement"],
            num_reps=optimization_options["ancilla_num_reps"],
            par_name="x",
        )
    elif optimization_options["ancilla_ansatz"] == "pma":
        ancilla_ansatz = pma(
            num_qubits=N,
            architecture=optimization_options["ancilla_ansatz_architecture"],
            num_reps=optimization_options["ancilla_num_reps"],
            par_name="x",
        )
    if optimization_options["system_ansatz"] == "two_local":
        system_ansatz = two_local(
            num_qubits=N,
            rotation_blocks=optimization_options["system_ansatz_rotation_blocks"],
            entanglement_blocks=optimization_options["system_ansatz_entanglement_blocks"],
            entanglement=optimization_options["system_ansatz_entanglement"],
            num_reps=optimization_options["system_num_reps"],
            par_name="y",
        )
    elif optimization_options["system_ansatz"] == "pma":
        system_ansatz = pma(
            num_qubits=N,
            architecture=optimization_options["system_ansatz_architecture"],
            num_reps=optimization_options["system_num_reps"],
            par_name="y",
        )
    return ancilla_ansatz, system_ansatz


def setup_backend(flag="statevector", model_tag=None):
    if flag == "statevector":
        backend = None
    elif flag == "qasm":
        backend = AerSimulator(method="statevector")
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
    H,
    flag="statevector",
    pma_flag=True,
    num_beta_points=4,
    path="./MHETS_data/",
    take_statevector_initial_parameters=False,
):
    if take_statevector_initial_parameters is True:
        # Takes optimized parameters from statevector simulation
        file_name = setup_file_name(H, flag="statevector", pma_flag=pma_flag)
        with open(path + file_name, "rb") as f:
            multi_beta_result_sv = pickle.load(f)
        initial_parameter_list = multi_beta_result_sv["optimized_parameter_list"]
        print("I AM TAKING INITIAL PARAMETER LIST FROM STATEVECTOR RESULTS")
    else:
        initial_parameter_list = [None for i in range(num_beta_points)]
    return initial_parameter_list


def setup_betas(old_betas, betas):
    new_betas = old_betas.copy()
    for beta in betas:
        try:
            index = new_betas.index(beta)
        except ValueError:
            new_betas.append(beta)
        else:
            if index < (len(new_betas) - 1):
                number_to_add = (beta + new_betas[index + 1]) / 2.0
                if number_to_add not in new_betas:
                    new_betas.append(number_to_add)
            else:
                number_to_add = beta + (beta - new_betas[index - 1]) / 2.0
                if number_to_add not in new_betas:
                    new_betas.append(number_to_add)
        new_betas.sort()
    return new_betas


def setup_file_name(H, flag="statevector", shots=1024, pma_flag=False):
    if flag == "statevector":
        file_name = ("MHETS_{}at_gy{}_B{}".format(H.N, H.gy, H.B).replace(".", "")) + ".pickle"
    elif flag == "qasm":
        file_name = (
            "MHETS_{}at_gy{}_B{}_{}shots".format(H.N, H.gy, H.B, shots).replace(".", "")
        ) + ".pickle"
    elif flag == "noise":
        file_name = (
            "MHETS_{}at_gy{}_B{}_{}shots_noise".format(H.N, H.gy, H.B, shots).replace(".", "")
        ) + ".pickle"
    elif flag == "hardware":
        file_name = (
            "MHETS_{}at_gy{}_B{}_{}shots_hardware".format(H.N, H.gy, H.B, shots).replace(".", "")
        ) + ".pickle"
    if pma_flag is True:
        file_name = file_name.replace(".pickle", "_pma.pickle")
    return file_name


def loading_offline_multi_data(path, file_names: list):
    multi_data = []
    for file_name in file_names:
        with open(path + file_name, "rb") as f:
            multi_data.append(pickle.load(f))
    return multi_data


def name_to_H(file_name):
    par_list = re.split(r"_", file_name)
    for string in par_list:
        if "at" in string:
            N = int(string.replace("at", ""))
        if "gy" in string:
            gy = float((string.replace("gy", "")).replace("0", "0.", 1))
        if "B" in string:
            B = float((string.replace("B", "")).replace("0", "0.", 1))
    return LMG_hamiltonian(N, gy, B)


def data_to_rho_s(data, H):
    ancilla_ansatz, system_ansatz = setup_ansatz(H.N, data["optimization_options"])
    mhets = MHETS_instance(
        H,
        ancilla_ansatz,
        system_ansatz,
        optimization_options=data["optimization_options"],
    )
    rho_s_list = []
    beta_list = data["betas"]
    for index in range(len(beta_list)):
        mhets.update_parameters(data["optimized_parameter_list"][index])
        rho = mhets.QST(mhets.build_total_circuit(), shots=data["optimization_options"]["shots"])
        rho_s_list.append(partial_trace(rho, range(ancilla_ansatz.get_num_qubits())))
    return rho_s_list
