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
import math

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

def qft_dagger(circ, bits):
    """n-qubit QFTdagger the first n qubits in circ"""
    n = len(bits)

    for i in range(n//2):
        circ.swap(bits[i], bits[n-i-1])

    for j in range(n):
        for m in range(j):
            circ.cu1(-math.pi/float(2**(j-m)), bits[m], bits[j])
        circ.h(j)

ampl_lst = [-0.81261, 0.171201, 0.16862325, -0.2227965, 0.171201, 0.12054625, 0.17434925, 0.04532175,
0.04532175, 0.165868, 0.12054625, - 0.2227965, 0.04532175, 0.04532175, 0.165868]

array = np.array(ampl_lst)
B = np.diag(array)

B_op = Operator(B)
print(B_op)



# This assumes that the counting qubits are the first qubits in the circuit
def phase_estimation(circ, main, counting, classical):
    n = len(counting)

    # Apply H-Gates to counting qubits
    circ.h(counting)

    # controlled-U operations
    angle = 2*math.pi / (2**n)
    repetitions = 1
    for bit in range(n):
        for i in range(repetitions):
            circ.cu1(angle, counting[bit], main[0]);
        repetitions *= 2

    # Do the inverse QFT:
    qft_dagger(circ, counting)

    # Measure
    for bit in range(n):
        circ.measure(counting[bit], classical[0])


main = QuantumRegister(4, 'main')
aux = QuantumRegister(4, 'auxiliary')
counting = QuantumRegister(2, 'counting')
classical = ClassicalRegister(10, 'classical')

circ = HartreeFock(some_args_here)
circ.add_register(aux)
circ.add_register(counting)
circ.add_register(classical)

phase_estimation(circ, main, counting, classical)
circ.draw('mpl')
plt.show()

# Hamiltonian = Operator([[-1.8310, 0.1813], [0.1813, -0.2537]])
# circ.unitary(Hamiltonian, q)
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
