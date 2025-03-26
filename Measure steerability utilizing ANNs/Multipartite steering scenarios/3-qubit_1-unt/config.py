import numpy as np
import tensorflow as tf
from targets import GHZ_assemblage_1unt_2m,GHZ_assemblage_1unt_3m

class Config:
    """ Config is just a collection of all metadata which can be accessed from auxiliary files as well. """
    def __init__(self):
        self.dim = 2 # Dimensionality of Hilbert-space.
        self.oa = self.dim # Number of outcomes.
        self.ma = 2 # Number of measurements, 2 or 3.
        # Visibility parameter of the target assemblage.
        self.visibility = (0, .1, .2, .3, .33, 1/3, .34, .4, .5, .6, .7, .8, .9, 1)  # GHZ_assemblage_1unt_2m
        # self.visibility = (0, .1, .2, .26, .2612, .27, .3, .4, .5, .6, .7, .8, .9, 1)  # GHZ_assemblage_1unt_3m
        
        # Neural network parameters
        self.lambda_outputsize = 1 + 2*4  # variables for \sigma_\lambda^B, \sigma_\lambda^C, and one for weight distribution.
        self.latin_depth = 1
        self.latin_width = 128
        self.nb_trains = 1  # choose how many times to train each target (numeric artifact that could be eliminated by increasing the number of trials per assemblage.)
        self.threshold_val = 0.003 # Threshold for stopping training
        self.Hidden_variables = (8,9)  # The cardinality of \lambda

        # Training procedure parameters
        self.no_of_batches = 4000  # How many batches to go through during training (iteration)
        self.weight_init_scaling = 3.  # Default is 1. Set to larger values to get more variance in initial weights.
        self.optimizer = 'adadelta'  # optimizer: 'adadelta', 'sgd', 'adam'
        self.learning_rate = 0.5
        self.decay = 0.001
        self.momentum = 0.25
        self.loss = 'td'  # 'td'  'qre'  'hs'
        self.no_of_validation_batches = 100  # How many batches to go through in each validation step. If batch size is large, 1 should be enough.
        self.activ = 'relu'  # 'relu'  'tanh' -- activation for most of NN
        self.activ2 = 'sigmoid'  # 'sigmoid'  'softmax' -- activation for last dense layer
        self.kernel_reg = None
        # self.activity_reg = 0.001

    def change_astarget(self, v):  # Change here the target assemblage imported from targets.py
        """set the target assemblage to a family defined in targets.py"""
        self.astarget = GHZ_assemblage_1unt_2m(v)
        self.asshape = tf.shape(self.astarget).numpy()

    def change_batch_size(self, H):
        self.input_lambda = H #  The cardinality of the hidden variable.
        self.inputsize= self.ma + H 
        self.batch_size = self.ma * H
        self.batch_size_test = self.batch_size  # In case we update batch_size we should also update the test batch size

def initialize():
    """ Initializes a Config class as a global variable pnn (for Parameters of Neural Network).
    The pnn object should be accessible and modifiable from all auxiliary files.
    """
    global pnn
    pnn = Config()