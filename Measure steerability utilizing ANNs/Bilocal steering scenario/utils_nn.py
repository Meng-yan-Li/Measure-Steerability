import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from scipy.stats import entropy
from tensorflow.keras.initializers import VarianceScaling
import time
import itertools
from tensorflow.keras.utils import plot_model

import config as cf

cf.initialize()
n = cf.pnn.dim  # Dimensionality of Hilbert-space

## Build ANN
def build_model():
    inputTensor = Input((cf.pnn.inputsize,))

    group_lambda_1 = Lambda(lambda x: x[:, :4], output_shape=((4,)))(inputTensor)
    group_lambda_2 = Lambda(lambda x: x[:, 4:8], output_shape=((4,)))(inputTensor)

    group_1 = group_lambda_1
    group_2 = Concatenate()([group_lambda_1, group_lambda_2])
    group_3 = group_lambda_2

    kernel_init = tf.keras.initializers.VarianceScaling(scale=cf.pnn.weight_init_scaling, mode='fan_in',distribution='truncated_normal', seed=None)
    
    for _ in range(cf.pnn.latin_depth):
        group_1 = Dense(cf.pnn.latin_width, activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg,
                        kernel_initializer=kernel_init)(group_1)
        group_2 = Dense(cf.pnn.latin_width*2, activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg,
                        kernel_initializer=kernel_init)(group_2)
        group_3 = Dense(cf.pnn.latin_width, activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg,
                        kernel_initializer=kernel_init)(group_3)

    # Apply final softmax layer
    group_1 = Dense(cf.pnn.lambda_outputsize, activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_1)
    group_2 = Dense(cf.pnn.b_outputsize, activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_2)
    group_3 = Dense(cf.pnn.lambda_outputsize, activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_3)
    outputTensor = Concatenate()([group_1, group_2, group_3])
    model = Model(inputTensor, outputTensor)
    return model

## Define the loss
def frobenius_norm(p, q):
    """Calculate the frobenius norm between two matrices p and q."""
    frobenius_norm = tf.norm(p-q)
    return frobenius_norm

def trace_distance(p, q):
    """Calculate the trace distance between two matrices p and q, with error handling."""
    e,_=tf.linalg.eigh(p-q)
    return  0.5*tf.math.reduce_sum(tf.math.abs(e),axis=-1) 

def qre(p, q):
    """quantum relative entropy, wher p,q represent matrix"""
    eigq, _ = tf.linalg.eigh(q)
    eigp, _ = tf.linalg.eigh(p)
    return tf.math.reduce_sum(eigq * (tf.math.log(eigq) - tf.math.log(eigp)))

def HS_distance(p, q):
    """Hilbert-Schmidt distance, wher p,q represent matrix"""
    return tf.math.sqrt(tf.linalg.trace(tf.linalg.matmul(tf.linalg.adjoint(p - q), p - q)))

def keras_distance(sigmap, sigmaq):
    """ Distance used in loss function. Where sigmap,sigmaq represent assemblage
        Implemented losses:
            fn:  frobenius norm
            td:  trace distance (recommended)
            qre: quantum relative entropy
            hs:  hilbert-Schmidt distance
    """
    _, _, b = cf.pnn.asshape
    distance_funcs = {
        'fn': frobenius_norm,
        'td': trace_distance,
        'qre': qre,
        'hs': HS_distance
    }
    loss_func = distance_funcs.get(cf.pnn.loss.lower())
    if loss_func is None:
        print('define the pnn.loss')
        return

    distances = tf.map_fn(lambda i: loss_func(sigmap[:, :, i], sigmaq[:, :, i]), 
                          tf.range(b, dtype=tf.int32), 
                          fn_output_signature=tf.float32)

    return tf.reduce_sum(distances)

def reduce_state(sigma):
    return tf.reduce_sum(sigma, axis=2)


## Construction of unsteerable state according to y_pred
def densitynK(arg):
    """Creates a density matrix for a pure state from 2n numbers."""

    # ## Mixed states
    # r = arg[0] # Bloch vector length (0 <= r <= 1)
    # theta = arg[1]*np.pi # Polar angle (0 <= theta <= pi)
    # phi = arg[2]*2*np.pi # Azimuthal angle (0 <= phi < 2*pi)
    # bloch = [
    #     r * tf.sin(theta) * tf.cos(phi),  # x-component
    #     r * tf.sin(theta) * tf.sin(phi),  # y-component
    #     r * tf.cos(theta)                # z-component
    # ]
    # bloch = tf.cast(bloch,dtype=tf.complex64)
    #
    # Pauli_x = tf.cast(np.array([[0, 1], [1, 0]]), 'complex64')
    # Pauli_y = tf.cast(np.array([[0, -1j], [1j, 0]]), 'complex64')
    # Pauli_z = tf.cast(np.array([[1, 0], [0, -1]]), 'complex64')
    #
    # return 0.5 * (tf.eye(2, dtype=tf.complex64) + bloch[0] * Pauli_x + bloch[1] * Pauli_y + bloch[2] * Pauli_z)

    # Pure states
    d = n
    arg = 2 * arg - 1 # Cast [0,1] --> [-1,1]
    norm = tf.sqrt(tf.reduce_sum(tf.square(arg)))

    # Create the pure state vector
    pure = [tf.complex(arg[2 * i] / norm, arg[2 * i + 1] / norm) for i in range(d)]
    pure = tf.reshape(tf.stack(pure), (d, 1))
    return tf.matmul(pure, pure, adjoint_b=True)

def integral(p_lambda_1, p_lambda_2, p_b_lambda12, sigma_lambda_1, sigma_lambda_2):
    """Combines into the NLHS model."""
    d1, d2, b = cf.pnn.asshape
    sigma_b = [
        tf.reduce_sum([
            p_lambda_1[i] * p_lambda_2[j] * p_b_lambda12[i * cf.pnn.cardinalities[1] + j, k] * tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(sigma_lambda_1[:, :, i]),tf.linalg.LinearOperatorFullMatrix(sigma_lambda_2[:, :, j])]).to_dense() for i in range(cf.pnn.cardinalities[0]) for j in range(cf.pnn.cardinalities[1])
        ], axis=0)
        for k in range(b)
    ]
    return tf.stack(sigma_b, axis=-1)

def customLoss_matrix(y_pred):
    """returns a assemblage of predictions"""
    lambda_1 = y_pred[:, :cf.pnn.lambda_outputsize]
    p_b_lambda12 = y_pred[:, cf.pnn.lambda_outputsize:cf.pnn.lambda_outputsize + cf.pnn.b_outputsize]
    lambda_2 = y_pred[:, cf.pnn.lambda_outputsize + cf.pnn.b_outputsize:]

    p_lambda_1 = tf.cast(lambda_1[::cf.pnn.cardinalities[1], 0], tf.complex64)
    arg_1 = lambda_1[::cf.pnn.cardinalities[1], 1:]

    p_lambda_2 = tf.cast(lambda_2[:cf.pnn.cardinalities[1], 0], tf.complex64)
    arg_2 = lambda_2[:cf.pnn.cardinalities[1], 1:]

    p_lambda_1 = p_lambda_1 / tf.reduce_sum(p_lambda_1)
    p_lambda_2 = p_lambda_2 / tf.reduce_sum(p_lambda_2)

    # Normalize p_b_lambda12 and cast to complex64
    p_b_lambda12 = tf.cast(p_b_lambda12 / tf.reduce_sum(p_b_lambda12, axis=1, keepdims=True), tf.complex64)

    sigma_lambda_1 = tf.stack([densitynK(tf.reshape(arg_1[i, :], (cf.pnn.lambda_outputsize - 1,))) for i in range(cf.pnn.cardinalities[0])],axis=-1)
    sigma_lambda_2 = tf.stack([densitynK(tf.reshape(arg_2[i, :], (cf.pnn.lambda_outputsize - 1,))) for i in range(cf.pnn.cardinalities[1])],axis=-1)

    return integral(p_lambda_1, p_lambda_2, p_b_lambda12, sigma_lambda_1, sigma_lambda_2),p_lambda_1, p_lambda_2, p_b_lambda12, sigma_lambda_1, sigma_lambda_2


## Loss function
def customLoss(y_true, y_pred):
    """return the loss of the ANN"""
    asshape = cf.pnn.asshape
    sigma_ture = tf.reshape(y_true[0,:], asshape)
    sigma_pred, _, _, _, _, _ = customLoss_matrix(y_pred)

    # # constrained optimization problem
    # reduce_sigma_true = reduce_state(sigma_ture) # \sum_b \sigma_{b} 
    # reduce_sigma_pred = reduce_state(sigma_pred) # \sum_b^{LHS} \sigma_{b} 
    # return tf.cond(tf.math.greater(trace_distance(reduce_sigma_true,reduce_sigma_pred),cf.pnn.threshold_val),
    #                 lambda: trace_distance(reduce_sigma_true,reduce_sigma_pred),
    #                 lambda: keras_distance(sigma_ture ,sigma_pred))
    
    return keras_distance(sigma_ture, sigma_pred)

def metric_td(y_true, y_pred):
    sigma_ture = cf.pnn.astarget
    sigma_pred, _, _, _, _, _ = customLoss_matrix(y_pred)
    asshape = cf.pnn.asshape
    td = tf.reduce_sum([trace_distance(sigma_ture[:, :, i], sigma_pred[:, :, i]) for i in range(asshape[2])])
    return td

## Generate the data sets
def generate_xy_batch():
    while True:
        # Generate combinations
        ranges = [range(1, v+1) for v in cf.pnn.cardinalities] 
        x_train = np.array(list(itertools.product(*ranges)))

        # Convert the values in each range into One-Hot codes of the corresponding lengths
        one_hot_encoded_columns = [np.eye(v)[x_train[:, i] - 1] for i, v in enumerate(cf.pnn.cardinalities)]

        # Splice the columns of the One-Hot code to form a complete matrix
        x_one_hot = np.hstack(one_hot_encoded_columns)

        asshape = cf.pnn.asshape
        y_true = tf.reshape(cf.pnn.astarget, [tf.math.reduce_prod(asshape)])
        y_true = tf.tile(tf.expand_dims(y_true, axis=0), [cf.pnn.batch_size, 1])  # Expand batch_size-fold
        yield (x_one_hot, y_true)


def generate_x_test():
    while True:
        # Generate combinations
        ranges = [range(1, v+1) for v in cf.pnn.cardinalities] 
        x_train = np.array(list(itertools.product(*ranges)))

        # Convert the values in each range into One-Hot codes of the corresponding lengths
        one_hot_encoded_columns = [np.eye(v)[x_train[:, i] - 1] for i, v in enumerate(cf.pnn.cardinalities)]

        # Splice the columns of the One-Hot code to form a complete matrix
        x_one_hot = np.hstack(one_hot_encoded_columns)
        yield x_one_hot

## Dynamically adjust the learning rate (not useful for 'Adadelta')
# class LRchanger(tf.keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         #  Change the learning rate at the start of epoch
#         if epoch!=0:
#             self.model.optimizer.lr=self.model.optimizer.lr*0.2
#             print(self.model.optimizer.lr.numpy())

## Callback function: Early stop
def CC_mindelta():
    """stop the training at the end of an epoch if the loss didn't decrease enough"""
    return tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=0, mode='auto',
                                            baseline=None, restore_best_weights=False)

class CC_minloss(tf.keras.callbacks.Callback):
    """stop the training at the end of an epoch if the loss is smaller than a value"""
    def on_epoch_end(self, batch, logs=None):
        if logs.get('val_metric_td') <= cf.pnn.threshold_val:
            tf.print('\n\n-----> stoping the training because the trace distance is smaller than ',cf.pnn.threshold_val)
            self.model.stop_training = True


## Training model
def single_run():
    """ Runs training algorithm for a single target assemblage. Returns model and data to fit."""
    # Model and optimizer related setup.
    K.clear_session()
    model = build_model()
    # plot_model(model,
    #     to_file='./figs/model.png',
    #     show_shapes=True,
    #     show_dtype=True,
    #     show_layer_names=True,
    #     rankdir='learning_rate',
    #     # expand_nested=True,
    #     dpi=200,
    #     show_layer_activations=True
    # )
    if cf.pnn.optimizer.lower() == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=cf.pnn.learning_rate, rho=0.95, epsilon=1e-7, decay=cf.pnn.decay)
    elif cf.pnn.optimizer.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=cf.pnn.learning_rate, decay=cf.pnn.decay, momentum=cf.pnn.momentum, nesterov=True)
    elif cf.pnn.optimizer.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=cf.pnn.learning_rate, beta_1=0.9, beta_2=0.999, decay=cf.pnn.decay, epsilon=1e-7)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=cf.pnn.learning_rate, decay=cf.pnn.decay, momentum=cf.pnn.momentum, nesterov=True)
        print(
            "\n\nWARNING!!! Optimizer {} not recognized. Please implement it if you want to use it. Using SGD instead.\n\n".format(
                cf.pnn.optimizer))
        cf.pnn.optimizer = 'sgd'  # set it for consistency.

    model.compile(loss=customLoss, optimizer=optimizer, metrics=[metric_td])

    # Fit model
    History = model.fit(generate_xy_batch(), steps_per_epoch=cf.pnn.no_of_batches, epochs=20, verbose=1,
                        callbacks=[CC_minloss(),CC_mindelta()], validation_data=generate_xy_batch(),
                        validation_steps=cf.pnn.no_of_validation_batches, class_weight=None, max_queue_size=20,
                        workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    return model, History

def single_evaluation_loss(model):
    """ Evaluates the model and returns the loss. """
    test_pred = model.predict(generate_x_test(), steps=1, verbose=0)
    # sigma_test_pred, p_lambda_1, p_lambda_2, p_b_lambda12, sigma_lambda_1, sigma_lambda_2 = customLoss_matrix(test_pred)
    sigma_test_pred, _, _, _, _, _ = customLoss_matrix(test_pred)
    sigma_target = cf.pnn.astarget
    asshape = cf.pnn.asshape

    # trace_distance
    loss = tf.reduce_sum([trace_distance(sigma_test_pred[:, :, i], sigma_target[:, :, i]) for i in range(asshape[2])])

    print("\n-----> Final evaluation loss:",loss.numpy(),"\n")
    
    # Save the approximate optimal data for one target assemblage
    # if loss.numpy() < cf.pnn.threshold_val:
    #     np.save('./data/p_lambda_1.npy', p_lambda_1.numpy())
    #     np.save('./data/p_lambda_2.npy', p_lambda_2.numpy())
    #     np.save('./data/p_b_lambda12.npy', p_b_lambda12.numpy())
    #     np.save('./data/sigma_lambda_1.npy', sigma_lambda_1.numpy())
    #     np.save('./data/sigma_lambda_2.npy', sigma_lambda_2.numpy())
    return loss