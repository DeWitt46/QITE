# -*- coding: utf-8 -*-
r"""
Class for tailoring TwoLocal ansatz



Created on Thu Dec  7 15:59:33 2023

@author: DeWitt
"""
# TODO: Add error raising
import math


from qiskit.circuit import QuantumCircuit, Parameter
from library.gate_creation import RYXGate, RXYGate


class two_local:
    def __init__(
        self,
        num_qubits: int,
        rotation_blocks: list[str] = ["ry"],
        entanglement_blocks: list[str] = ["cx"],
        entanglement: str = "linear",
        num_reps: int = 1,
        par_name: str = "x",
    ):
        self.num_qubits = num_qubits
        self.rotation_blocks = rotation_blocks
        self.entanglement_blocks = entanglement_blocks
        self.entanglement = entanglement
        self.num_reps = num_reps
        self.par_name = par_name
        self.par = [
            Parameter("{}_{}".format(par_name, i))
            for i in range(
                0, self.num_qubits * len(self.rotation_blocks) * (self.num_reps + 1)
            )
        ]
        self.circuit = QuantumCircuit(self.num_qubits)

    def get_name(self):
        return "two_local"

    def get_circuit(self):
        return self.circuit

    def get_num_qubits(self):
        return self.num_qubits

    def get_rotation_blocks(self):
        return self.rotation_blocks

    def get_entanglement_blocks(self):
        return self.entanglement_blocks

    def get_entanglement(self):
        return self.entanglement

    def get_num_reps(self):
        return self.num_reps

    def get_par_name(self):
        return self.par_name

    def get_parameters(self):
        return self.par

    def get_num_parameters(self):
        return len(self.par)

    def add_entanglement_layer(self, qc: QuantumCircuit, entanglement: str):
        if entanglement == "linear":
            for qubit in range(0, qc.num_qubits - 1):
                qc.cx(qubit, qubit + 1)
        elif entanglement == "all":
            for control_qubit in range(0, qc.num_qubits - 1):
                for control_target_distance in range(1, qc.num_qubits - control_qubit):
                    qc.cx(control_qubit, control_qubit + control_target_distance)

    def add_rotation_layer(self, qc, rotation_blocks, par):
        for i in range(0, qc.num_qubits):
            for gate in rotation_blocks:
                if gate == "rx":
                    qc.rx(par[i], i)
                elif gate == "ry":
                    qc.ry(par[i], i)
                elif gate == "rz":
                    qc.rz(par[i], i)

    def bind_parameters(self, par: list):
        self.par = par

    def build(self):
        self.circuit = QuantumCircuit(self.num_qubits)
        for rep in range(self.num_reps):
            self.add_rotation_layer(
                self.circuit,
                self.rotation_blocks,
                self.par[
                    self.num_qubits
                    * len(self.rotation_blocks)
                    * (rep) : self.num_qubits
                    * len(self.rotation_blocks)
                    * (rep + 1)
                ],
            )
            self.add_entanglement_layer(self.circuit, self.entanglement)
        self.add_rotation_layer(
            self.circuit,
            self.rotation_blocks,
            self.par[self.num_qubits * len(self.rotation_blocks) * (self.num_reps) :],
        )
        return self.circuit


class pma:
    r"""
        Class to define a Phisically Motivated Ansatz for the LMG model: RP(x_i, x_j) 
        rotation blocks made of 2-qubit gates RP(x_i, x_j) = RXY(x_i)RYX(x_j).
        RP preserves Parity.
        The RP gates can be arranged in a linear architecture, or full architecture (Two-Local fashion)
    """

    def __init__(
        self,
        num_qubits: int,
        architecture: str = "linear",
        num_reps: int = 1,
        par_name: str = "x",
    ):
        self.num_qubits = num_qubits
        self.architecture = architecture
        self.num_reps = num_reps
        self.par_name = par_name
        if self.architecture == "full":
            self.par = [
                Parameter("{}_{}".format(self.par_name, i))
                for i in range(
                    0, 2 * math.comb(self.num_qubits, 2) * (self.num_reps + 1)
                )
            ]
        elif self.architecture == "linear":
            self.par = [
                Parameter("{}_{}".format(self.par_name, i))
                for i in range(0, 2 * (self.num_qubits - 1) * (self.num_reps + 1))
            ]
        self.circuit = QuantumCircuit(self.num_qubits)

    def get_name(self):
        return "pma"

    def get_circuit(self):
        return self.circuit

    def get_num_qubits(self):
        return self.num_qubits

    def get_architecture(self):
        return self.architecture

    def get_num_reps(self):
        return self.num_reps

    def get_par_name(self):
        return self.par_name

    def get_parameters(self):
        return self.par

    def get_num_parameters(self):
        return len(self.par)

    def add_layer(self, qc: QuantumCircuit, architecture: str, rep: int):
        if architecture == "linear":
            # Take a portion of the parameter list to use for this layer
            partial_par = self.par[
                rep * 2 * (self.num_qubits - 1) : (rep + 1) * 2 * (self.num_qubits - 1)
            ]
            par_counter = 0
            for qubit in range(0, qc.num_qubits - 1):
                qc.append(
                    RXYGate(partial_par[par_counter]), [qubit, qubit + 1], [],
                )
                qc.append(
                    RYXGate(partial_par[par_counter + 1]), [qubit, qubit + 1], [],
                )
                par_counter += 1
        elif architecture == "full":
            # Take a portion of the parameter list to use for this layer
            partial_par = self.par[
                rep
                * 2
                * math.comb(self.num_qubits, 2) : (rep + 1)
                * 2
                * math.comb(self.num_qubits, 2)
            ]
            par_counter = 0
            for first_qubit in range(0, qc.num_qubits - 1):
                for first_second_distance in range(1, qc.num_qubits - first_qubit):
                    qc.append(
                        RXYGate(partial_par[par_counter]),
                        [first_qubit, first_qubit + first_second_distance],
                        [],
                    )
                    qc.append(
                        RYXGate(partial_par[par_counter + 1]),
                        [first_qubit, first_qubit + first_second_distance],
                        [],
                    )
                    par_counter += 1

    def bind_parameters(self, par: list):
        self.par = par

    def build(self):
        self.circuit = QuantumCircuit(self.num_qubits)
        for rep in range(self.num_reps + 1):
            self.add_layer(qc=self.circuit, architecture=self.architecture, rep=rep)
        return self.circuit
