from typing import Union, List, Optional, Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.operators import LegacyBaseOperator, Z2Symmetries
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from .q_equation_of_motion import QEquationOfMotion

def diffuser(circuit):
    """Apply inversion about the average step of Grover's algorithm."""
    """ three qubits version of the S operator
    qubits = circuit.qubits
    nqubits = len(qubits)
    
    for q in range(nqubits):
        circuit.h(q)
        circuit.x(q)
    
    # Do controlled-Z
    circuit.h(2)
    circuit.ccx(0,1,2)
    circuit.h(2)
    
    for q in range(nqubits):
        circuit.x(q)
        circuit.h(q)
