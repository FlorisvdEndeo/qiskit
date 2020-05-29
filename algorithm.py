import numpy as np
import math
import pickle

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, BasicAer
from qiskit.visualization import plot_histogram
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity
from qiskit.providers import ibmq

import qiskit.extensions
from qiskit.chemistry.components.initial_states import HartreeFock
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

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


def s_operator(circ, aux):
    circ.h(aux)
    circ.x(aux)
    circ.h(aux[3])
    circ.ccx(aux[0],aux[4],aux[3])
    circ.ccx(aux[1],aux[3],aux[2])

ampl_lst = [-0.81261, 0.171201, 0.16862325, -0.2227965, 0.171201, 0.12054625, 0.17434925, 0.04532175,
0.04532175, 0.165868, 0.12054625, - 0.2227965, 0.04532175, 0.04532175, 0.165868]
aux_state = Initialize(ampl_lst)
aux_state.definition
B_op = Operator(B)
array = np.array(ampl_lst)
B = np.diag(array)

main = QuantumRegister(5, 'main')
aux = QuantumRegister(4, 'auxiliary')
counting = QuantumRegister(2, 'counting')
classical = ClassicalRegister(10, 'classical')

circ = HartreeFock(2, 1, 'parity').construct_circuit('circuit', main)
circ.add_register(aux)
circ.add_register(counting)
circ.add_register(classical)
circ.append(B_op, aux)
phase_estimation(circ, main, counting, classical)
circ.draw('mpl')
plt.show()

# Hamiltonian = Operator([[-1.8310, 0.1813], [0.1813, -0.2537]])
# circ.unitary(Hamiltonian, q)
#circ.append(Hamiltonian)
#circ.decompose().draw()
#Hamiltonian.IsUnitary()


### Running the code at IBM
# backend = ibmq.least_busy()
# qobj = assemble(transpile(circ, backend=backend), backend=backend)
# job = backend.run(qobj)
# result = job.result(refresh=True)
# with open('results.pkl', 'wb') as f:
#     pickle.dumps(result)
# plot_histogram(result.get_counts())
# plt.show()
