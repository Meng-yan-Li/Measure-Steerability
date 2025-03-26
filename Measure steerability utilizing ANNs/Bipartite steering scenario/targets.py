import tensorflow as tf
import numpy as np

# Function to generate Isotropic assemblage
def Isotropic_assemblage(d, v, mubs=None):
    """
    `Isotropic_assemblage` generates Isotropic Assemblage.

    Args:
        d (int) - Dimensionality of the corresponding system.
        v (int) - Visibility.
        mubs (int) - Mutual unbiased bases.
    Returns:
        sigma: Subnormalized states (assemblage) prepared for the trusted party.
    """
    if mubs is None:
        if d == 4:
            mubs = load_mubs_4()  
        else:
            mubs = MUBs(d)

    sigma = np.zeros((d, d, d, d + 1), dtype=complex)
    for x in range(d + 1):
        for a in range(d):
            sigma[:, :, a, x] = (
                v / d * np.dot(mubs[x][a], mubs[x][a].T.conj()).T + (1 - v) / (d**2) * np.eye(d)
            )
    return tf.cast(sigma,dtype=tf.complex64)

# Function to generate Isotropic assemblage with 2 measurements
def Isotropic_assemblage_2m(d, v, mubs=None):
    if mubs is None:
        if d == 4:
            mubs = load_mubs_4()[:2]  # Load only first two MUBs.
        else:
            mubs = MUBs(d)[:2]

    sigma = np.zeros((d, d, d, 2), dtype=complex)
    for x in range(2):
        for a in range(d):
            sigma[:, :, a, x] = (
                v / d * np.dot(mubs[x][a], mubs[x][a].T.conj()).T + (1 - v) / (d**2) * np.eye(d)
            )
    return tf.cast(sigma,dtype=tf.complex64)

# Function to generate Isotropic assemblage with 3 measurements
def Isotropic_assemblage_3m(d, v, mubs=None):
    if mubs is None:
        if d == 4:
            mubs = load_mubs_4()[:3]  # Load only first three MUBs.
        else:
            mubs = MUBs(d)[:3]

    sigma = np.zeros((d, d, d, 3), dtype=complex)
    for x in range(3):
        for a in range(d):
            sigma[:, :, a, x] = (
                v / d * np.dot(mubs[x][a], mubs[x][a].T.conj()).T + (1 - v) / (d**2) * np.eye(d)
            )
    return tf.cast(sigma,dtype=tf.complex64)

# Function to load MUBs for d=4
def load_mubs_4():
    mubs = [
        [
            np.array([[1], [0], [0], [0]], dtype=complex),
            np.array([[0], [1], [0], [0]], dtype=complex),
            np.array([[0], [0], [1], [0]], dtype=complex),
            np.array([[0], [0], [0], [1]], dtype=complex),
        ],
        [
            0.5 * np.array([[1], [1], [1], [1]], dtype=complex),
            0.5 * np.array([[1], [1], [-1], [-1]], dtype=complex),
            0.5 * np.array([[1], [-1], [-1], [1]], dtype=complex),
            0.5 * np.array([[1], [-1], [1], [-1]], dtype=complex),
        ],
        [
            0.5 * np.array([[1], [-1], [-1j], [-1j]], dtype=complex),
            0.5 * np.array([[1], [-1], [1j], [1j]], dtype=complex),
            0.5 * np.array([[1], [1], [1j], [-1j]], dtype=complex),
            0.5 * np.array([[1], [1], [-1j], [1j]], dtype=complex),
        ],
        [
            0.5 * np.array([[1], [-1j], [-1j], [-1]], dtype=complex),
            0.5 * np.array([[1], [-1j], [1j], [1]], dtype=complex),
            0.5 * np.array([[1], [1j], [1j], [-1]], dtype=complex),
            0.5 * np.array([[1], [1j], [-1j], [1]], dtype=complex),
        ],
        [
            0.5 * np.array([[1], [-1j], [-1], [-1j]], dtype=complex),
            0.5 * np.array([[1], [-1j], [1], [1j]], dtype=complex),
            0.5 * np.array([[1], [1j], [-1], [1j]], dtype=complex),
            0.5 * np.array([[1], [1j], [1], [-1j]], dtype=complex),
        ],
    ]
    return mubs

# Function to generate MUBs for general d
def MUBs(d):
    if not is_prime(d) or d == 2:
        if np.log2(d).is_integer():
            FourierBase = []
            CompuBase = []
            for x in range(d):
                OneHot = np.zeros((d, 1))
                OneHot[x] = 1
                Base_x = []
                for a in range(d):
                    Base_a = np.array([
                        np.exp(1j * np.pi / 2 * absolute_trace(x + 2 * a, 2, int(np.log2(d))) * l)
                        for l in range(d)
                    ])[:, None]
                    Base_a /= np.sqrt(d)
                    Base_x.append(Base_a)
                CompuBase.append(OneHot)
                FourierBase.append(Base_x)
            CompuBase = [CompuBase]
            return CompuBase + FourierBase
    else:
        FourierBase = []
        CompuBase = []
        for x in range(d):
            OneHot = np.zeros((d, 1))
            OneHot[x] = 1
            Base_x = []
            for a in range(d):
                Base_a = np.array([
                    np.exp(1j * 2 * np.pi / d * (a * l + x * l**2))
                    for l in range(d)
                ])[:, None]
                Base_a /= np.sqrt(d)
                Base_x.append(Base_a)
            CompuBase.append(OneHot)
            FourierBase.append(Base_x)

        CompuBase = [CompuBase]
        return CompuBase + FourierBase

# Utility function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Function to compute the absolute trace from Fq to Fp
def absolute_trace(x, p, m):
    """
    Computes the absolute trace from Fq to Fp.

    Args:
        x (int) - An element of Fq (given as an integer representation).
        p (int) - The characteristic of the prime field Fp.
        m (int) - The degree of the field extension Fq = Fp^m.

    Returns:
        trace (int) - The absolute trace of x in Fp.
    """
    trace = 0
    for i in range(m):
        trace += x**(p**i)
    return trace
