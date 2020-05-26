# qiskit

Interesting literature:

https://github.com/Qiskit/qiskit-tutorials/blob/9405254b38312771f8d5c2dd6f451cec35307995/tutorials/simulators/1_aer_provider.ipynb


The algorithm can be found in the method.pdf

The Hamiltonian can be found in generalcircuit.pdf on page 8

Implementation of H_2 with simpler hamiltonian in Hydrogen Nmr.pdf


The algorithms protocol: 

Three registers

Auxillary for Phase Estimation

Ancillary which we prepare in \ket{\beta}: four qubits, if we pick the bigger Hamiltonian

Physical register, which we prepare in an approximation of the eigenstate of the Hamiltonian. Might use something like this: https://github.com/Qiskit/qiskit-aqua/blob/master/qiskit/chemistry/applications/molecular_ground_state_energy.py

We need to create the S and V operator.

S is just the Grover Diffusion operator: See  https://qiskit.org/textbook/ch-algorithms/grover.html

V is applying Pj to the jth state of the ancillary qubit: if our physical system has 4 qubits, then a 16-dimensional vector describes the state. then V applies the P_j to the jth position in that state. So then 

``` python
def diffuser(circuit):
    """Apply inversion about the average step of Grover's algorithm."""
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
```
