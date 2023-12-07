# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:59:33 2023

@author: DeWitt
"""
from qiskit.circuit import QuantumCircuit, Parameter


class two_local:
    def __init__(
        self,
        num_qubit: int,
        rotation_blocks: list[str] = ["ry"],
        entanglement_blocks: list[str] = ["cx"],
        entanglement: str = "linear",
        num_rep: int = 1,
        par_name: str = "x",
    ):
        self.num_qubit = num_qubit
        self.rotation_blocks = rotation_blocks
        self.entanglement_blocks = entanglement_blocks
        self.entanglement = entanglement
        self.num_rep = num_rep
        self.par_name = par_name
        self.par = [
            Parameter("{}_{}".format(par_name, i))
            for i in range(0, num_qubit * len(rotation_blocks) * (num_rep + 1))
        ]
        self.circuit = QuantumCircuit(self.num_qubit)

    def add_entanglement_layer(self, qc : QuantumCircuit = self.circuit, entanglement : str = self.entanglement):
        if entanglement == "linear":
            for qubit in range(0, qc.num_qubit - 1):
                qc.circuit.cx(qubit, qubit + 1)

    def add_rotation_layer(self, qc = self.circuit, rotation_blocks = self.rotation_blocks, par : list = self.par):
        for i in range(0, qc.num_qubits):
            for gate in rotation_blocks:
                if gate == "rx":
                    qc.rx(par[i], i)
                elif gate == "ry":
                    qc.ry(par[i], i)
                elif gate == "rz":
                    qc.rz(par[i], i)
                    
    def bind_parameters(self, par : list):
        self.par = par

    def build(self):
        add_rotation_layer(self)
        for rep in range(num_rep):
            

    # for rep in range(1, reps + 1):
    #     if entanglement == "linear":
    #         lin_entanglement(qc)
    #     single_rep(qc, gates, par[n * len(gates) * (rep) : n * len(gates) * (rep + 1)])
    # return qc

    def get_circuit(self):
        return self.circuit

    def get_num_qubit(self):
        return self.num_qubit

    def get_rotation_blocks(self):
        return self.rotation_blocks

    def get_entanglement_blocks(self):
        return self.entanglement_blocks

    def get_entanglement(self):
        return self.entanglement

    def get_num_rep(self):
        return self.num_rep

    def get_par_name(self):
        return self.par_name

    def get_parameters(self):
        return self.par

def()
