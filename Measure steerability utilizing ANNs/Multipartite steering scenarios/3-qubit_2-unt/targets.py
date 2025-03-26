import numpy as np
import tensorflow as tf

# Function to generate GHZ assemblage **(2-UNT & 2-M)**
def GHZ_assemblage_2unt_2m(w):
    """
    TThe assemblage created for Charlie after Alice and Bob each performs two Pauli spin measurements
    (X and Z) on her/his particle of the three-qubit GHZ state.
    """
    Pauli_x = np.array([[0, 1], [1, 0]])
    Pauli_z = np.array([[1, 0], [0, -1]])
    sigma = np.zeros((2, 2, 2, 2, 2, 2), dtype=complex)
    diagonal = np.diag([(1 - w) / 8, (1 - w) / 8])
    Pauli = [Pauli_x, Pauli_z]
    for x in range(2):
        for a in range(2):
            for y in range(2):
                for b in range(2):
                    projector_A = (np.eye(2) + ((-1) ** a) * Pauli[x]) / 2
                    projector_B = (np.eye(2) + ((-1) ** b) * Pauli[y]) / 2
                    sigma[:, :, a, b, x, y] = diagonal + (w / 2) * np.array([
                        [projector_A[0, 0] * projector_B[0, 0], projector_A[1, 0] * projector_B[1, 0]],
                        [projector_A[0, 1] * projector_B[0, 1], projector_A[1, 1] * projector_B[1, 1]]
                    ])
    return tf.cast(sigma,'complex64')

# Function to generate GHZ assemblage **(2-UNT & 3-M)**
def GHZ_assemblage_2unt_3m(w):
    """
    TThe assemblage created for Charlie after Alice and Bob each performs three Pauli spin measurements
    (X, Y and Z) on her/his particle of the three-qubit GHZ state.
    """
    Pauli_x = np.array([[0, 1], [1, 0]])
    Pauli_y = np.array([[0, -1j], [1j, 0]])
    Pauli_z = np.array([[1, 0], [0, -1]])
    sigma = np.zeros((2, 2, 2, 2, 3, 3), dtype=complex)
    diagonal = np.diag([(1 - w) / 8, (1 - w) / 8])
    Pauli = [Pauli_x, Pauli_y, Pauli_z]
    for x in range(3):
        for a in range(2):
            for y in range(3):
                for b in range(2):
                    projector_A = (np.eye(2) + ((-1) ** a) * Pauli[x]) / 2
                    projector_B = (np.eye(2) + ((-1) ** b) * Pauli[y]) / 2
                    sigma[:, :, a, b, x, y] = diagonal + (w / 2) * np.array([
                        [projector_A[0, 0] * projector_B[0, 0], projector_A[1, 0] * projector_B[1, 0]],
                        [projector_A[0, 1] * projector_B[0, 1], projector_A[1, 1] * projector_B[1, 1]]
                    ])
    return tf.cast(sigma,'complex64')