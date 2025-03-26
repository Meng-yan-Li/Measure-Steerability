import numpy as np
import tensorflow as tf

# Function to generate GHZ assemblage **(1-UNT & 2-M)**
def GHZ_assemblage_1unt_2m(v):
    """
    The assemblage created for Bob and Charlie after Alice performs two Pauli spin measurements
    (X and Z) on her particle of the three-qubit GHZ state.
    """
    Pauli_x = np.array([[0, 1], [1, 0]])
    Pauli_z = np.array([[1, 0], [0, -1]])
    sigma = np.zeros((4, 4, 2, 2), dtype=complex)
    diagonal = np.diag([(1 - v) / 8, (1 - v) / 8, (1 - v) / 8, (1 - v) / 8])
    Pauli = [Pauli_x, Pauli_z]
    for x in range(2):  # Loop over x = 0, 1 (corresponding to X, Z measurements)
        for a in range(2):  # Loop over a = 0, 1 (two possible measurement outcomes)
            projector_A = (np.eye(2) + ((-1) ** a) * Pauli[x]) / 2
            sigma[:, :, a, x] = diagonal + (v / 2) * np.array([
                        [projector_A[0, 0], 0, 0, projector_A[1, 0]],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [projector_A[0, 1], 0, 0, projector_A[1, 1]]])
    return tf.cast(sigma,'complex64')

# Function to generate GHZ ssemblage **(1-UNT & 3-M)**
def GHZ_assemblage_1unt_3m(v):
    """
    TThe assemblage created for Bob and Charlie after Alice performs three Pauli spin measurements
    (X, Y and Z) on her particle of the three-qubit GHZ state.
    """
    Pauli_x = np.array([[0, 1], [1, 0]])
    Pauli_y = np.array([[0, -1j], [1j, 0]])
    Pauli_z = np.array([[1, 0], [0, -1]])
    sigma = np.zeros((4, 4, 2, 3), dtype=complex)
    diagonal = np.diag([(1 - v) / 8, (1 - v) / 8, (1 - v) / 8, (1 - v) / 8])
    Pauli = [Pauli_x, Pauli_y, Pauli_z]
    for x in range(3):  # Loop over x = 0, 1 ,2 (corresponding to X, Y, Z measurements)
        for a in range(2):  # Loop over a = 0, 1 (two possible measurement outcomes)
            projector_A = (np.eye(2) + ((-1) ** a) * Pauli[x]) / 2
            sigma[:, :, a, x] = diagonal + (v / 2) * np.array([
                        [projector_A[0, 0], 0, 0, projector_A[1, 0]],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [projector_A[0, 1], 0, 0, projector_A[1, 1]]])
    return tf.cast(sigma,'complex64')
    

# Function to generate W assemblage **(1-UNT & 2-M)**
def W_assemblage_1unt_2m(v):
    """
    The assemblage created for Bob and Charlie after Alice performs two Pauli spin measurements
    (X and Z) on her particle of the three-qubit GHZ state.
    """
    Pauli_x = np.array([[0, 1], [1, 0]])
    Pauli_z = np.array([[1, 0], [0, -1]])
    sigma = np.zeros((4, 4, 2, 2), dtype=complex)
    diagonal = np.diag([(1 - v) / 8, (1 - v) / 8, (1 - v) / 8, (1 - v) / 8])
    Pauli = [Pauli_x, Pauli_z]
    for x in range(2):  # Loop over x = 0, 1 (corresponding to X, Z measurements)
        for a in range(2):  # Loop over a = 0, 1 (two possible measurement outcomes)
            projector_A = (np.eye(2) + ((-1) ** a) * Pauli[x]) / 2
            sigma[:, :, a, x] = diagonal + (v / 3) * np.array([
                        [projector_A[1, 1], projector_A[0, 1], projector_A[0, 1], 0],
                        [projector_A[1, 0], projector_A[0, 0], projector_A[0, 0], 0],
                        [projector_A[1, 0], projector_A[0, 0], projector_A[0, 0], 0],
                        [0, 0, 0, 0]])
    return tf.cast(sigma,'complex64')

# Function to generate W assemblage **(1-UNT & 3-M)**
def W_assemblage_1unt_3m(v):
    """
    TThe assemblage created for Bob and Charlie after Alice performs three Pauli spin measurements
    (X, Y and Z) on her particle of the three-qubit GHZ state.
    """
    Pauli_x = np.array([[0, 1], [1, 0]])
    Pauli_y = np.array([[0, -1j], [1j, 0]])
    Pauli_z = np.array([[1, 0], [0, -1]])
    sigma = np.zeros((4, 4, 2, 3), dtype=complex)
    diagonal = np.diag([(1 - v) / 8, (1 - v) / 8, (1 - v) / 8, (1 - v) / 8])
    Pauli = [Pauli_x, Pauli_y, Pauli_z]
    for x in range(3):  # Loop over x = 0, 1 ,2 (corresponding to X, Y, Z measurements)
        for a in range(2):  # Loop over a = 0, 1 (two possible measurement outcomes)
            projector_A = (np.eye(2) + ((-1) ** a) * Pauli[x]) / 2
            sigma[:, :, a, x] = diagonal + (v / 3) * np.array([
                        [projector_A[1, 1], projector_A[0, 1], projector_A[0, 1], 0],
                        [projector_A[1, 0], projector_A[0, 0], projector_A[0, 0], 0],
                        [projector_A[1, 0], projector_A[0, 0], projector_A[0, 0], 0],
                        [0, 0, 0, 0]])
    return tf.cast(sigma,'complex64')