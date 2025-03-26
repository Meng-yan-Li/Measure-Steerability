import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras.regularizers import L2
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
    
    kernel_init = tf.keras.initializers.VarianceScaling(scale=cf.pnn.weight_init_scaling, mode='fan_in',
                                                        distribution='truncated_normal', seed=None)
    group_lambda = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(inputTensor)
    for _ in range(cf.pnn.latin_depth):
        group_lambda = Dense(cf.pnn.latin_width, activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg,
                        kernel_initializer=kernel_init)(group_lambda)

    # Apply final softmax layer
    group_lambda = Dense(cf.pnn.lambda_outputsize, activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_lambda)
    outputTensor = group_lambda
    model = Model(inputTensor, outputTensor)
    return model


## Define the loss
def frobenius_norm(p, q):
    """Calculate the frobenius norm between two matrices p and q."""
    frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(p - q)))
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
    """ Distance used in loss function. Where sigmap,sigmaq represent assemblage sigma_{a|x} = [:,:,a,x]"""
    # some distances defined here are not usable for training
    asshape = cf.pnn.asshape
    distance_fn_map = {
        'fn': frobenius_norm,
        'td': trace_distance,
        'qre': qre,
        'hs': HS_distance
    }
    distance_fn = distance_fn_map.get(cf.pnn.loss.lower())
    if distance_fn is None:
        print('Define the pnn.loss')
    else:
        loss_sum = tf.reduce_sum([
            distance_fn(sigmap[:, :, i, j, k, l], sigmaq[:, :, i, j, k, l])
            for l in range(asshape[-1]) for k in range(asshape[-2]) for j in range(asshape[-3]) for i in range(asshape[-4])
        ])
        return loss_sum/np.prod(cf.pnn.settings[:,0]) 

def reduce_state(sigma):
    return tf.reduce_sum(sigma, axis=[2,3])


## Construction of unsteerable state according to y_pred
def densitynK(arg):
    """Creates a density matrix for a pure state from 2n numbers."""
    d = n  # Dimensionality of the system

    ## Bloch vector representation
    # if d == 2: # Bloch vector
    #     # Linear mapping
    #     theta = tf.cast(arg[0] * (tf.constant(np.pi) / 2),dtype=tf.complex64)
    #     phi = tf.cast(arg[1] * tf.constant(np.pi),dtype=tf.complex64)

    #     a = tf.math.cos(theta / 2)
    #     b = tf.exp(1j * phi) * tf.math.sin(theta / 2)

    #     # Create the pure state vector
    #     pure = tf.reshape(tf.stack([a, b]), (d, 1))
    #     density_matrix = tf.matmul(pure, pure, adjoint_b=True)
    # else:
    arg = 2 * arg - 1
    norm = tf.sqrt(tf.reduce_sum(tf.square(arg)))

    # Create the pure state vector
    pure = [tf.complex(arg[2 * i] / norm, arg[2 * i + 1] / norm) for i in range(d)]
    pure = tf.reshape(tf.stack(pure), (d, 1))
    return tf.matmul(pure, pure, adjoint_b=True)

def integral(p_lambda, sigma_lambda):
    """Combines into the MLHS_2 model."""
    d1, d2, a, b, x, y = cf.pnn.asshape
    D_A = tf.cast(cf.pnn.Det_strategy[0],dtype=tf.complex64)
    D_B = tf.cast(cf.pnn.Det_strategy[1],dtype=tf.complex64)
    sigma = tf.zeros([d1, d2, a, b, x, y], dtype=tf.complex64)
    for l in range(y):
        for k in range(x):
            for j in range(b):
                for i in range(a):
                    vec_a = D_A[i,k,:]
                    vec_b = D_B[j,l,:]
                    D = tf.reshape(vec_a[:, tf.newaxis] * vec_b[tf.newaxis, :], [-1]) # Use the broadcast mechanism
                    sigma_abxy = tf.math.reduce_sum(D * p_lambda * sigma_lambda, axis=-1)
                    indices = tf.constant([[m, n, i, j, k ,l] for m in range(d1) for n in range(d2)], dtype=tf.int32)
                    updates = tf.reshape(sigma_abxy, [-1])
                    sigma = tf.tensor_scatter_nd_update(sigma, indices, updates)
    return sigma

def customLoss_matrix(y_pred):
    """return the p_lambda,sigma_lambda from y_pred"""
    p = y_pred[:,0]
    p_lambda = tf.cast(p / tf.math.reduce_sum(p),dtype=tf.complex64)
    arg = y_pred[:,1:]
    sigma_lambda = tf.stack([densitynK(tf.reshape(tf.slice(arg, (i, 0), (1, cf.pnn.lambda_outputsize - 1)), (cf.pnn.lambda_outputsize - 1,))) for i in range(cf.pnn.batch_size)], axis=-1)  # transfer list to tensor
    return integral(p_lambda, sigma_lambda)


## Loss function
def customLoss(y_true, y_pred):
    """return the loss of the ANN"""
    asshape = cf.pnn.asshape
    sigma_ture = tf.reshape(y_true[0,:], asshape)
    sigma_pred = customLoss_matrix(y_pred)

    # reduce_sigma_true = reduce_state(sigma_ture)
    # reduce_sigma_pred = reduce_state(sigma_pred)

    # # constrained optimization problem
    # reduce_loss = tf.reduce_sum([trace_distance(reduce_sigma_true[:,:,i,j],reduce_sigma_pred[:,:,i,j]) for j in range(asshape[-1]) for i in range(asshape[-2])])
    # return tf.cond(tf.math.greater(reduce_loss,0.005*(asshape[-2]*asshape[-1])),
    #             lambda: reduce_loss ,
    #             lambda: keras_distance(sigma_ture ,sigma_pred))

    return keras_distance(sigma_ture ,sigma_pred)
    
# def customLoss_eval(y_pred):
#     """loss of a prediction, not use in training loop"""
#     sigma_ture = cf.pnn.astarget
#     sigma_pred = customLoss_matrix(y_pred)
#     return keras_distance(sigma_ture,sigma_pred)

def metric_td(y_true, y_pred):
    sigma_ture = cf.pnn.astarget
    sigma_pred = customLoss_matrix(y_pred)
    asshape = cf.pnn.asshape
    td = tf.reduce_sum([trace_distance(sigma_ture[:, :, i, j, k, l], sigma_pred[:, :, i, j, k, l])
            for l in range(asshape[-1]) for k in range(asshape[-2]) for j in range(asshape[-3]) for i in range(asshape[-4])])
    return td/np.prod(cf.pnn.settings[:,0]) 


## Generate the data sets
def generate_xy_batch():
    while True:
        x_train = np.array([i for i in range(cf.pnn.inputsize)])
        x_train = tf.one_hot(x_train, depth=cf.pnn.inputsize) 

        asshape = cf.pnn.asshape
        y_true = tf.reshape(cf.pnn.astarget, [tf.math.reduce_prod(asshape)])
        y_true = tf.tile(tf.expand_dims(y_true, axis=0), [cf.pnn.batch_size, 1])  # Expand batch_size-fold
        yield (x_train, y_true)

def generate_x_test():
    while True:
        x_train = np.array([i for i in range(cf.pnn.inputsize)])
        x_train = tf.one_hot(x_train, depth=cf.pnn.inputsize) 
        yield (x_train,)

## Dynamically adjust the learning rate 
# class LRchanger(tf.keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         #  Change the learning rate at the start of epoch
#         if epoch!=0:
#             self.model.optimizer.lr=self.model.optimizer.lr*0.2
#             print(self.model.optimizer.lr.numpy())

## Callback function: Early stop
def CC_mindelta():
    """stop the training at the end of an epoch if the loss didn't decrease enough"""
    return tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='auto',
                                            baseline=None, restore_best_weights=False)

class CC_minloss(tf.keras.callbacks.Callback):
    """stop the training at the end of an epoch if the loss is smaller than a value"""
    def on_epoch_end(self, batch, logs=None):
        if logs.get('val_metric_td') <= cf.pnn.threshold_val:
            tf.print('\n\n-----> stopping the training because the trace distance is smaller than', cf.pnn.threshold_val)
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
    # model.summary()

    # Fit model
    History = model.fit(generate_xy_batch(), steps_per_epoch=cf.pnn.no_of_batches, epochs=20, verbose=1,
                        callbacks=[CC_minloss(), CC_mindelta()], validation_data=generate_xy_batch(),
                        validation_steps=cf.pnn.no_of_validation_batches, class_weight=None, max_queue_size=20,
                        workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)

    return model, History

def single_evaluation_loss(model):
    """ Evaluates the model and returns the loss. """
    test_pred = model.predict(generate_x_test(), steps=1, verbose=0)
    sigma_test_pred = customLoss_matrix(test_pred)
    sigma_target = cf.pnn.astarget
    asshape = cf.pnn.asshape

    # trace_distance
    loss = tf.reduce_sum([trace_distance(sigma_target[:, :, i, j, k, l], sigma_test_pred[:, :, i, j, k, l])
            for l in range(asshape[-1]) for k in range(asshape[-2]) for j in range(asshape[-3]) for i in range(asshape[-4])])/np.prod(cf.pnn.settings[:,0]) 
    
    print("\n-----> Final evaluation loss:",loss.numpy(),"\n")
    return loss