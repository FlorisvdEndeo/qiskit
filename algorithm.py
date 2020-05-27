# from typing import Union, List, Optional, Callable
# from qiskit.providers import BaseBackend
# from qiskit.aqua import QuantumInstance
# from qiskit.aqua.algorithms import VQE
# from qiskit.aqua.operators import LegacyBaseOperator, Z2Symmetries
# from qiskit.aqua.components.optimizers import Optimizer
# from qiskit.aqua.components.variational_forms import VariationalForm
# from qiskit.aqua.utils.validation import validate_min, validate_in_set
#
#
# import numpy as np
#
# from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
# from qiskit import execute, BasicAer
# from qiskit.compiler import transpile
# from qiskit.quantum_info.operators import Operator, Pauli
# from qiskit.quantum_info import process_fidelity
#
# from qiskit.extensions import RXGate, XGate, CXGate
# from qiskit import *
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity

from qiskit.extensions import RXGate, XGate, CXGate

# HH2 =−0.81261I+0.171201σ0z+0.16862325σ1z−0.2227965σ2z
# + 0.171201σ1zσ0z +0.12054625σ2zσ0z +0.17434925σ3zσ1z
# + 0.04532175σ2xσ1zσ0x +0.04532175σ2yσ1zσ0y + 0.165868σ2zσ1zσ0z
# + 0.12054625σzσzσz − 0.2227965σzσzσz + 0.04532175σzσxσzσx 320 321 3210
# +0.04532175σ3zσ2yσ1zσ0y +0.165868σ3zσ2zσ1zσ0z

#pauli_ops = [−0.81261*Operator(Pauli(label='I')), 0.171201*Operator(Pauli(label='Z')), 0.16862325*


Hamiltonian = Operator([[-1.8310, 0.1813], [0.1813, -0.2537]])
q = QuantumRegister(3, 'q')
circ = QuantumCircuit(q)
circ.unitary(Hamiltonian, q)
#circ.append(Hamiltonian)
#circ.decompose().draw()
#Hamiltonian.IsUnitary()


# def diffuser(circuit):
#     """Apply inversion about the average step of Grover's algorithm."""
#     """ three qubits version of the S operator"""
#     qubits = circuit.qubits
#     nqubits = len(qubits)
#
#     for q in range(nqubits):
#         circuit.h(q)
#         circuit.x(q)
#
#     # Do controlled-Z
#     circuit.h(2)
#     circuit.ccx(0,1,2)
#     circuit.h(2)
#
#     for q in range(nqubits):
#         circuit.x(q)
#         circuit.h(q)
