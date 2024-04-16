# -*- coding: utf-8 -*-
r"""
Class for tailoring gates not implemented in Qiskit



Created on Fri Apr  5 18:07:44 2024

@author: DeWitt
"""

import math
import numpy as np
from typing import Optional
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType


# pylint: disable=cyclic-import
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.rz import RZGate


class RXYGate(Gate):
    r"""A parametric 2-qubit :math:`X \otimes Y` interaction (rotation about XY).
    
    """

    def __init__(
        self,
        theta: ParameterValueType,
        label: Optional[str] = None,
        *,
        duration=None,
        unit="dt"
    ):
        """Create new RXY gate."""
        super().__init__("rxy", 2, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        """
        Calculate a subcircuit that implements this unitary.
        """

        #      ┌─────────┐                   ┌──────────┐
        # q_0: ┤    H    ├──■─────────────■──┤     H    ├
        #      ├─────────┤┌─┴─┐┌───────┐┌─┴─┐├──────────┤
        # q_1: ┤ Rx(π/2) ├┤ X ├┤ Rz(0) ├┤ X ├┤ Rx(-π/2) ├
        #      └─────────┘└───┘└───────┘└───┘└──────────┘

        theta = self.params[0]
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[0]], []),
            (RXGate(np.pi / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[0]], []),
            (RXGate(-np.pi / 2), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return inverse RXY gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.XYGate` with an inverted parameter value.

         Returns:
            XYGate: inverse gate.
        """
        return RXYGate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the XY gate."""
        import numpy

        half_theta = float(self.params[0]) / 2
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        return numpy.array(
            [[cos, 0, 0, -sin], [0, cos, -sin, 0], [0, sin, cos, 0], [sin, 0, 0, cos],],
            dtype=dtype,
        )

    def power(self, exponent: float):
        """Raise gate to a power."""
        (theta,) = self.params
        return RXYGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RXYGate):
            return self._compare_parameters(other)
        return False


class RYXGate(Gate):
    r"""A parametric 2-qubit :math:`Y \otimes X` interaction (rotation about YX).
    
    """

    def __init__(
        self,
        theta: ParameterValueType,
        label: Optional[str] = None,
        *,
        duration=None,
        unit="dt"
    ):
        """Create new RYX gate."""
        super().__init__("ryx", 2, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        """
        Calculate a subcircuit that implements this unitary.
        """

        #      ┌─────────┐                   ┌──────────┐
        # q_0: ┤ Rx(π/2) ├──■─────────────■──┤ Rx(-π/2) ├
        #      ├─────────┤┌─┴─┐┌───────┐┌─┴─┐├──────────┤
        # q_1: ┤    H    ├┤ X ├┤ Rz(0) ├┤ X ├┤    H     ├
        #      └─────────┘└───┘└───────┘└───┘└──────────┘

        theta = self.params[0]
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RXGate(np.pi / 2), [q[0]], []),
            (HGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RXGate(-np.pi / 2), [q[0]], []),
            (HGate(), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return inverse RYX gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.XYGate` with an inverted parameter value.

         Returns:
            YXGate: inverse gate.
        """
        return RYXGate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the YX gate."""
        import numpy

        half_theta = float(self.params[0]) / 2
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        return numpy.array(
            [[cos, 0, 0, -sin], [0, cos, sin, 0], [0, -sin, cos, 0], [sin, 0, 0, cos],],
            dtype=dtype,
        )

    def power(self, exponent: float):
        """Raise gate to a power."""
        (theta,) = self.params
        return RXYGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RYXGate):
            return self._compare_parameters(other)
        return False
