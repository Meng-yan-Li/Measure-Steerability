import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
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

    group_x = Lambda(lambda x: x[:,:cf.pnn.ma], output_shape=((cf.pnn.ma,)))(inputTensor)
    group_lambda = Lambda(lambda x: x[:,cf.pnn.ma:], output_shape=((cf.pnn.input_lambda,)))(inputTensor)

    group_1 = Concatenate()([group_x,group_lambda])
    group_2 = group_lambda

    kernel_init = tf.keras.initializers.VarianceScaling(scale=cf.pnn.weight_init_scaling, mode='fan_in', distribution='truncated_normal', seed=None)
    for _ in range(cf.pnn.latin_depth):
        group_1 = Dense(2*cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_1)
        group_2 = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_2)
        
    # Apply final softmax layer
    group_1 = Dense(cf.pnn.oa,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_1)
    group_2 = Dense(cf.pnn.lambda_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_2)
    outputTensor = Concatenate()([group_1,group_2])
    model = Model(inputTensor,outputTensor)
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
            distance_fn(sigmap[:, :, k, i], sigmaq[:, :, k, i])
            for i in range(asshape[3]) for k in range(asshape[2])
        ])
        return loss_sum/cf.pnn.ma

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

def integral(p_a_x_lambda,p_lambda,sigma_lambda):
    """Combines into the MLHS_1 model."""
    d1, d2, a, x = cf.pnn.asshape
    sigma = tf.zeros([d1, d2, a, x], dtype=tf.complex64)
    for i in range(x):
        for k in range(a):
            sigma_ax = tf.reduce_sum(tf.reshape(p_a_x_lambda[i*cf.pnn.input_lambda:(i+1)*cf.pnn.input_lambda,k], (cf.pnn.input_lambda,))*p_lambda*sigma_lambda ,axis=-1) 
            indices = tf.constant([[y, z, k, i] for y in range(d1) for z in range(d2)], dtype=tf.int32)
            updates = tf.reshape(sigma_ax, [-1])
            # Update the tensor
            sigma = tf.tensor_scatter_nd_update(sigma, indices, updates)
    return sigma

def customLoss_matrix(y_pred):
    """return the p_a_x_lambda,p_lambda,sigma_lambda from y_pred"""
    p_a_x_lambda = y_pred[:,:n]
    p_a_x_lambda = tf.cast(p_a_x_lambda / tf.reduce_sum(p_a_x_lambda,axis=1, keepdims=True),dtype=tf.complex64)

    p_lambda = y_pred[:cf.pnn.input_lambda,n:n+1]
    p_lambda = tf.cast(p_lambda / tf.reduce_sum(p_lambda),dtype=tf.complex64)
    p_lambda = tf.reshape(p_lambda, (cf.pnn.input_lambda,))

    rho = y_pred[:cf.pnn.input_lambda,n+1:]

    sigma_lambda_b = tf.stack([densitynK(tf.reshape(rho[i,:(cf.pnn.lambda_outputsize - 1) // 2], ((cf.pnn.lambda_outputsize - 1) // 2,))) for i in range(cf.pnn.input_lambda)], axis=-1)
    sigma_lambda_c = tf.stack([densitynK(tf.reshape(rho[i,(cf.pnn.lambda_outputsize - 1) // 2:], ((cf.pnn.lambda_outputsize - 1) // 2,))) for i in range(cf.pnn.input_lambda)], axis=-1)

    sigma_lambda = tf.stack([tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(densitynK(tf.reshape(rho[i,:(cf.pnn.lambda_outputsize - 1) // 2], ((cf.pnn.lambda_outputsize - 1) // 2,)))),tf.linalg.LinearOperatorFullMatrix(densitynK(tf.reshape(rho[i,(cf.pnn.lambda_outputsize - 1) // 2:], ((cf.pnn.lambda_outputsize - 1) // 2,))))]).to_dense() for i in range(cf.pnn.input_lambda)], axis=-1) # \sigma_{\lambda_1} \otimes \sigma_{\lambda_2} 

    return integral(p_a_x_lambda,p_lambda,sigma_lambda),p_a_x_lambda,p_lambda,sigma_lambda_b,sigma_lambda_c


## Loss function
def customLoss(y_true, y_pred):
    """return the loss of the ANN"""
    asshape = cf.pnn.asshape
    sigma_ture = tf.reshape(y_true[0,:], asshape)
    sigma_pred,_,_,_,_ = customLoss_matrix(y_pred)

    # # Constrained optimization problem
    # reduce_sigma_true = reduce_state(sigma_ture) # \sum_a \sigma_{a|x}
    # reduce_sigma_pred = reduce_state(sigma_pred) # \sum_a^{LHS} \sigma_{a|x}
    # reduce_loss = tf.reduce_sum([trace_distance(reduce_sigma_true[:,:,i],reduce_sigma_pred[:,:,i]) for i in range(asshape[3])])
    # return tf.cond(tf.math.greater(reduce_loss, cf.pnn.threshold_val*asshape[3]),
    #                             lambda: reduce_loss,
    #                             lambda: keras_distance(sigma_ture ,sigma_pred))

    return keras_distance(sigma_ture, sigma_pred)

def metric_td(y_true, y_pred):
    sigma_ture = cf.pnn.astarget
    sigma_pred,_,_,_,_ = customLoss_matrix(y_pred)
    asshape = cf.pnn.asshape
    td = tf.reduce_sum([trace_distance(sigma_ture[:, :, j, i], sigma_pred[:, :, j, i]) for i in range(asshape[3]) for j in range(asshape[2])])
    return td/cf.pnn.ma


## Generate the data sets
def generate_xy_batch():
    while True:
        # Generate combinations
        x = tf.eye(cf.pnn.ma, dtype=tf.int32)  
        l = tf.eye(cf.pnn.input_lambda, dtype=tf.int32)  
        rows1 = tf.unstack(x)
        rows2 = tf.unstack(l)
        cartesian_rows = [
            tf.concat([row1, row2], axis=0)  
            for row1, row2 in itertools.product(rows1, rows2)
        ]
        x_one_hot = tf.stack(cartesian_rows) 

        asshape = cf.pnn.asshape
        y_true = tf.reshape(cf.pnn.astarget, [tf.math.reduce_prod(asshape)])
        y_true = tf.tile(tf.expand_dims(y_true, axis=0), [cf.pnn.batch_size, 1])  # Expand batch_size-fold
        yield (x_one_hot, y_true)


def generate_x_test():
    while True:
        # Generate combinations
        x = tf.eye(cf.pnn.ma, dtype=tf.int32)  
        l = tf.eye(cf.pnn.input_lambda, dtype=tf.int32)  
        rows1 = tf.unstack(x)
        rows2 = tf.unstack(l)
        cartesian_rows = [
            tf.concat([row1, row2], axis=0)  
            for row1, row2 in itertools.product(rows1, rows2)
        ]
        x_one_hot = tf.stack(cartesian_rows) 
        yield x_one_hot
        
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

    # Fit model
    History = model.fit(generate_xy_batch(), steps_per_epoch=cf.pnn.no_of_batches, epochs=20, verbose=1,
                        callbacks=[CC_minloss()], validation_data=generate_xy_batch(),
                        validation_steps=cf.pnn.no_of_validation_batches, class_weight=None, max_queue_size=20,
                        workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)

    return model, History

def single_evaluation_loss(model):
    """ Evaluates the model and returns the loss. """
    test_pred = model.predict(generate_x_test(), steps=1,max_queue_size=20,workers=1, use_multiprocessing=False, verbose=0)
    # sigma_test_pred,p_a_x_lambda,p_lambda,sigma_lambda_b,sigma_lambda_c = customLoss_matrix(test_pred)
    sigma_test_pred,_,_,_,_ = customLoss_matrix(test_pred)
    sigma_target = cf.pnn.astarget
    asshape = cf.pnn.asshape

    # trace_distance
    loss = tf.reduce_sum([trace_distance(sigma_test_pred[:, :, j, i], sigma_target[:, :, j, i]) for i in range(asshape[3]) for j in range(asshape[2])])/cf.pnn.ma
    
    print("\n-----> Final evaluation loss:",loss.numpy(),"\n")

    # np.save('./data/p_a_x_lambda.npy', p_a_x_lambda.numpy())
    # np.save('./data/p_lambda.npy', p_lambda.numpy())
    # np.save('./data/sigma_lambda_b.npy', sigma_lambda_b.numpy())
    # np.save('./data/sigma_lambda_c.npy', sigma_lambda_c.numpy())
    return loss