import numpy as np
import tensorflow as tf

# Function to generate Isotropic assemblage
def Bilocal_Iso_assemblage_1m_sym(v):
    """
    The assemblage created for Alice and Charlie after Bob performs a Bell state measurement.
    The two Isotropic states have the same visibility.
    """
    bell_states = tf.constant([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1, -1, 0]], dtype=tf.complex64) / tf.sqrt(tf.constant(2.0, dtype=tf.complex64))
    bell_states = tf.stack([tf.matmul(tf.reshape(bell_states[i,:], (4, 1)), tf.reshape(bell_states[i,:], (4, 1)), adjoint_b=True) for i in range(4)],axis=-1)
    sigma = tf.stack([((v**2) / 4.0) * bell_states[:,:,i] + ((1-v**2)/16.0) * np.eye(4, dtype=np.complex64) for i in range(4)],axis=-1)
    return sigma # Convert back to a TensorFlow tensor

def Bilocal_Iso_assemblage_1m_asy(v,w):
    """
    The assemblage created for Alice and Charlie after Bob performs a Bell state measurement.
    The two Isotropic states have different visibilities.
    """
    bell_states = tf.constant([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1, -1, 0]], dtype=tf.complex64) / tf.sqrt(tf.constant(2.0, dtype=tf.complex64))
    bell_states = tf.stack([tf.matmul(tf.reshape(bell_states[i,:], (4, 1)), tf.reshape(bell_states[i,:], (4, 1)), adjoint_b=True) for i in range(4)],axis=-1)
    sigma = tf.stack([((w*v) / 4.0) * bell_states[:,:,i] + ((1-w*v)/16.0) * np.eye(4, dtype=np.complex64) for i in range(4)],axis=-1)
    return sigma # Convert back to a TensorFlow tensor