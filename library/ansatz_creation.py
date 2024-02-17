# -*- coding: utf-8 -*-
r"""
Class for tailoring TwoLocal ansatz



Created on Thu Dec  7 15:59:33 2023

@author: DeWitt
"""
# TODO: Add error raising
from qiskit.circuit import QuantumCircuit, Parameter


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
            for i in range(0, num_qubits * len(rotation_blocks) * (num_reps + 1))
        ]
        self.circuit = QuantumCircuit(self.num_qubits)

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
