# -*- coding: utf-8 -*-
r"""
Collection of functions that other classes use to work with labels of product operators or statevectors.
The point is that It's easier to write labels in the script and let the code do the converting.



Created on Tue Dec 12 15:44:02 2023

@author: DeWitt
"""
import numpy as np
from itertools import product


from qiskit import QuantumCircuit
from qiskit.primitives import Sampler, Estimator


operator_allowed = ["x", "z"]
op_to_basis_state_dict = {"x": ["-", "+"], "z": ["0", "1"]}
basis_state_to_par_dict = {"-": -np.pi / 2, "+": np.pi / 2, "0": 0, "1": np.pi}


def generate_basis_measure_list(N, operators=["x"], flag="all_possible"):
    r"""Generates the product operators labels you want to measure to get the chain.

    Three ways to use this function are allowed:
        -Generate all combinations of the single qubit operators provided;
        -Generate product operators that are the tensor product of the single qubit operators provided;
        -Use the product operators provided.

    .. code-block::python
        from library import state_label as lb

        N = 3

        # If you want to generate all possible combinations of single qubit operators
        op1 = ["x", "z"]
        flag1 = "all_possible"

        # If you want to generate product operators that are the product of the same single qubit operator
        op2 = ["xx", "xz"]
        flag2 = "every_qbit_same"

        # If you want to choose the product operators to measure by yourself
        op3 = ["xzx", "zzz", "xxz"]
        flag3 = "manual"

        print("First way:", generate_basis_measure_list(N, op1, flag1))
        print("Second way:", generate_basis_measure_list(N, op2, flag2))
        print("Third way:", generate_basis_measure_list(N, op3, flag3))

    Args:
        N: Number of Qubits.
        operators: List of single qubit operators or list of product operators.
        flag: String that defines the way the function works. Flags allowed are 'all_possible', 'every_qbit_same' or 'manual'

    Returns:
        List of product operators labels.
    """
    if flag == "all_possible":
        basis_measure_list = []
        for i in product(operators, repeat=N):
            basis_measure_list.append("".join(i))
    elif flag == "every_qbit_same":
        basis_measure_list = []
        for op in operators:
            basis_measure_list.append(op * N)
    elif flag == "manual":
        basis_measure_list = operators
    else:
        print(
            "flags allowed are 'all_possible', 'every_qbit_same' or 'manual'"
        )
        print(
            "For the latter, you have to insert strings like [xxx, zxz, ecc]"
        )
    return basis_measure_list


def generate_basis_list(N, operators=["x"], flag="all_possible"):
    r"""Generates the basis statevectors labels of the product operators QMETTS intends to measure to get the chain.

    Three ways to use this function are allowed:
        -Basis of product operators generated as all the combinations of the single qubit operator provided;
        -Basis of product operators that are the tensor product of the single qubit operators provided;
        -Basis of the product operators provided.

    .. code-block::python
        from library import state_label as lb

        N = 3

        # If you want the basis of product operators generated  as all the combinations of single qubit operators
        op1 = ["x", "z"]
        flag1 = "all_possible"

        # If you want the basis of product operators that are the product of the same single qubit operator
        op2 = ["xx", "xz"]
        flag2 = "every_qbit_same"

        # If you want the basis of chosen product operators
        op3 = ["xzx", "zzz", "xxz"]
        flag3 = "manual"

        print("First way:", generate_basis_list(N, op1, flag1))
        print("Second way:", generate_basis_list(N, op2, flag2))
        print("Third way:", generate_basis_list(N, op3, flag3))

    Args:
        N: Number of Qubits.
        operators: List of single qubit operators or list of product operators.
        flag: String that defines the way the function works. Flags allowed are 'all_possible', 'every_qbit_same' or 'manual'

    Returns:
        List of basis statevectors labels.
    """
    if flag == "all_possible":
        basis_list = []
        basis_dict = []
        for op in operators:
            for state in op_to_basis_state_dict[op]:
                basis_dict.append(state)
        for i in product(basis_dict, repeat=N):
            basis_list.append("".join(i))
    elif flag == "every_qbit_same":
        for op in operators:
            for i in product(op_to_basis_state_dict[op], repeat=N):
                basis_list.append("".join(i))
    elif flag == "manual":
        basis_list = []
        for string in operators:
            for i in product(["0", "1"], repeat=len(string)):
                order = "".join(i)
                basis_state = ""
                for pos in range(len(order)):
                    basis_state += op_to_basis_state_dict[string[pos]][
                        int(order[pos])
                    ]
                basis_list += [basis_state]
    else:
        print(
            "flags allowed are 'all_possible', 'every_qbit_same' or 'manual'"
        )
        print(
            "For the latter, you have to insert strings like [xxx, zxz, ecc]"
        )
    return basis_list


def state_to_par(label, num_params=6):
    r"""Converts the statevector label provided to a set of ansatz parameters.

        [!!! So far only two_local ansatz has been tackled !!!]

    Args:
        label: String that defines a statevector. [Single qubit labels allowed are "+","-","0","1"]
        num_params: Number of the anstaz parameters.

    Returns:
        List of ansatz parameters.
    """
    par_list = np.zeros(num_params)
    for pos in range(len(label)):
        par_list[pos - len(label)] = basis_state_to_par_dict[label[pos]]
    return par_list


def random_measure_label(basis_measure_list: list):
    r"""Randomly chooses a product operator among a provided list.

    Args:
        basis_measure_list: List of product operators to choose from.

    Returns:
        A string defining the chosen product operator.
    """
    n = len(basis_measure_list)
    choice = np.random.randint(n)
    measure_label = basis_measure_list[choice]
    return measure_label


def measure_label_to_circuit(qc, measure_label: str):
    r"""Adds the product operator measure to the circuit provided.

    The function reads the label which defines the product operator to measure and adds the appropriate gates to the circuit provided.

    Args:
        qc: Starting QiskitCircuit.
        measure_label: String that defines the product operator to measure on the state evolved by the starting circuit.

    Returns:
        QiskitCircuit with implemented measures.
    """
    n_qbit = qc.num_qubits
    new_qc = QuantumCircuit(n_qbit)
    new_qc.append(qc, list(np.arange(n_qbit)))
    for basis_pos in range(len(measure_label)):
        if measure_label[basis_pos] == "x":
            new_qc.h(basis_pos)
        elif measure_label[basis_pos] == "z":
            pass
        else:
            print("Wrong basis to measure into")
    new_qc.measure_all()
    return new_qc


def z_to_x_state_label(state: str, measure_label: str):
    r"""Converts the result of a product operator measure to the appropriate statevector label.

    On a IBM hardware measures are made in Z basis, so If your goal was to measure in a different basis, you have to convert the result obtained to the appropriate label.

    Args:
        state: Quasi-state measured.
        measure_label: String that defines the product operator's label.

    Returns:
        Label of the appropriate statevector the measure made the initial statevector collapse to.
    """
    x_state = ""
    for i in range(len(state)):
        qbit = list(state)[i]
        if list(measure_label)[i] == "x":
            if qbit == "0":
                x_state += "+"
            elif qbit == "1":
                x_state += "-"
        elif list(measure_label)[i] == "z":
            x_state += qbit
    return x_state


def choose_state(circuit, basis_measure_list):
    r"""Measures a random product operator to collapse the initial statevector to a random one.

    The function copies the circuit provided, which evolved |0> into a certain statevector, and measures a random product operator (among the ones in the provided list) once to collapse the state into a random one.

    Args:
        circuit: QiskitCircuit that evolves |0> into a certain statevector.
        basis_measure_list: List of product operators you want to measure.

    Returns:
        Label of a random statevector, which is an eigenvector of the product operators provided.
    """
    measure_label = random_measure_label(basis_measure_list)
    new_qc = measure_label_to_circuit(circuit, measure_label)
    sampler = Sampler()
    job = sampler.run(new_qc, shots=1)
    temporary_state_label = list(
        job.result().quasi_dists[0].binary_probabilities().keys()
    )[0]
    chosen_state_label = z_to_x_state_label(
        state=temporary_state_label, measure_label=measure_label
    )
    return chosen_state_label


def exp_value(circuit, observable, estimator=Estimator()):
    r"""Computes the expectation value of an observable on a certain statevector.

    Args:
        circuit: QiskitCircuit which evolves |0> to a certain statevector, on which you want the observable expectation value to be computed.
        observable: SparsePauliOp of the observable you want the expectation value.
        estimator: Qiskit estimator (Only used the one in qiskit.primitives so far).

    Returns:
        Expectation value of the provided observable on the provided statevector.
    """
    job = estimator.run(circuit, observable)
    return job.result().values[0]
