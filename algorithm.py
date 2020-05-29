import numpy as np
import math
import pickle
import argparse

from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, IBMQ, execute, BasicAer
from qiskit.visualization import plot_histogram
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity
from qiskit.providers import ibmq
from qiskit.extensions import Initialize
from qiskit.chemistry.components.initial_states import HartreeFock

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
        circ.h(bits[j])


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
            circ.cu1(angle, counting[bit], main[0])
        repetitions *= 2

    # Do the inverse QFT:
    qft_dagger(circ, counting)

    # Measure
    for bit in range(n):
        circ.measure(counting[bit], classical[bit])


def s_operator(circ, aux):
    circ.h(aux)
    circ.x(aux)
    circ.h(aux[3])
    circ.ccx(aux[0],aux[4],aux[3])
    circ.ccx(aux[1],aux[3],aux[2])


ampl_lst = [-0.81261, 0.171201, 0.16862325, -0.2227965, 0.171201, 0.12054625, 0.17434925, 0.04532175,
0.04532175, 0.165868, 0.12054625, - 0.2227965, 0.04532175, 0.04532175, 0.165868, 0.0000]
array = np.asarray(ampl_lst,  dtype=np.float)

newarray = array.reshape(1,-1)
normalizedhm = preprocessing.normalize(newarray, norm='l2')
normalized = normalizedhm[0]

def full_circuit():
    """This is the full circuit that can be expanded into the full algorithm"""
    main = QuantumRegister(5, 'main')
    aux = QuantumRegister(4, 'auxiliary')
    counting = QuantumRegister(2, 'counting')
    classical = ClassicalRegister(10, 'classical')
    circ = HartreeFock(2, 1, 'parity').construct_circuit('circuit', main)
    circ.add_register(aux)
    circ.add_register(counting)
    circ.add_register(classical)

    instruction_set = circ.initialize(normalized, aux)
    phase_estimation(circ, main, counting, classical)

    # Does not have a stardard label
    # Also the params mess up the drawing
    instruction_set[0].label = 'init'
    params = instruction_set[0].params
    instruction_set[0].params = []
    circ.draw('mpl')
    plt.savefig('full_schematic.png')
    plt.show()
    instruction_set[0].params = params


def real_circuit():
    """Run a smaller circuit on a real computer"""
    # This circuit is smaller so we can run it at IBM
    main = QuantumRegister(1, 'main')
    counting = QuantumRegister(2, 'counting')
    classical = ClassicalRegister(10, 'classical')
    circ = QuantumCircuit(main, counting, classical)
    phase_estimation(circ, main, counting, classical)
    circ.draw('mpl')
    plt.savefig('simple_schematic.png')
    return

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    provider.backends()
    backend = ibmq.least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 3 and
                                      not b.configuration().simulator and b.status().operational==True))

    job = execute(circ, backend=backend, shots=8000)
    result = job.result()
    with open('results.pkl', 'wb') as f:
        pickle.dumps(result)
    plot_histogram(result.get_counts())
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run some quantum code')
    parser.add_argument(
        '-r', '--real',
        help='Run on a real quantum computer',
        action='store_true'
    )
    args = parser.parse_args()
    if args.real:
        real_circuit()
    else:
        full_circuit()

# Hamiltonian = Operator([[-1.8310, 0.1813], [0.1813, -0.2537]])
# circ.unitary(Hamiltonian, q)
#circ.append(Hamiltonian)
#circ.decompose().draw()
#Hamiltonian.IsUnitary()
