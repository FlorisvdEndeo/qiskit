# from qiskit.providers import BaseBackend
# from qiskit.aqua import QuantumInstance
# from qiskit.aqua.algorithms import VQE
# from qiskit.aqua.operators import LegacyBaseOperator, Z2Symmetries
# from qiskit.aqua.components.optimizers import Optimizer
# from qiskit.aqua.components.variational_forms import VariationalForm
# from qiskit.aqua.utils.validation import validate_min, validate_in_set
#
# from qiskit import *
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity

from qiskit.extensions import RXGate, XGate, CXGate
import matplotlib.pyplot as plt

# One of the possible hamiltonians we could use:
# HH2 = −0.81261I + 0.171201σ0z + 0.16862325σ1z − 0.2227965σ2z
#      + 0.171201σ1zσ0z + 0.12054625σ2zσ0z + 0.17434925σ3zσ1z
#      + 0.04532175σ2xσ1zσ0x + 0.04532175σ2yσ1zσ0y + 0.165868σ2zσ1zσ0z
#      + 0.12054625σzσzσz − 0.2227965σzσzσz + 0.04532175σzσxσzσx
#      + 0.04532175σ3zσ2yσ1zσ0y + 0.165868σ3zσ2zσ1zσ0z

def parse_pauli_spec(spec):
    op = Operator(Pauli(label=spec[0]))
    for i in range(1,4):
        op = op.tensor(Pauli(label=spec[i]))
    return op

pauli_specs = [
    'IIII',
    'ZIII',
    'IZII',
    'IIZI',
    'ZZII',
    'ZIZI',
    'IZIZ',
    'XZXI',
    'YZYI',
    'ZZZI',
    'ZIZZ',
    'IZZZ',
    'XZXZ',
    'YZYZ',
    'ZZZZ',
]

def qft_dagger(circ, n):
    """n-qubit QFTdagger the first n qubits in circ"""

    for qubit in range(n//2):
        circ.swap(qubit, n-qubit-1)
   
    for j in range(n):
        for m in range(j):
            circ.cu1(-math.pi/float(2**(j-m)), m, j)
        circ.h(j)


# This assumes that the counting qubits are the first qubits in the circuit
def phase_estimation(circ, counting_qubits):
    # Apply H-Gates to counting qubits
    for qubit in range(counting_qubits):
        circ.h(qubit)

    # Prepare our eigenstate |psi>
    circ.x(counting_qubits)

    # controlled-U operations
    angle = 2*math.pi / (2**counting_qubits)
    repetitions = 1
    for bit in range(counting_qubits):
        for i in range(repetitions):
            circ.cu1(angle, bit, counting_qubits);
        repetitions *= 2

    # Do the inverse QFT:
    qft_dagger(circ, counting_qubits)

    # Measure
    for bit in range(counting_qubits):
        circ.measure(bit,bit)


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
