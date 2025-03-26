import numpy as np
import tensorflow as tf 
from math import sqrt
from targets import Bilocal_Iso_assemblage_1m_sym, Bilocal_Iso_assemblage_1m_asy

class Config:
    """ Config is just a collection of all metadata which can be accessed from auxiliary files as well. """
    def __init__(self):
        self.dim = 2  # Dimensionality of Hilbert-space.
        self.inputsize= 2 * 4 # \lambda_1, \lambda_2
        self.b_outputsize = 4 # The central party performs one 4-outcome measurement.
        self.visibility = (0, .1, .2, .3, .4, .5, 1/sqrt(3), .6, .7, .8, .9, 1)

        # Neural network parameters
        self.lambda_outputsize = 5 # variables for \rho_\lambda, and one for weight distribution
        self.latin_depth = 1
        self.latin_width = 128
        self.nb_trains = 1  # choose how many times to train each target (numeric artifact that could be eliminated by increasing the number of trials per assemblage.)
        self.threshold_val = 0.001 # Threshold for stopping training
        self.Hidden_variables = ([4,4],) # [\Lambda_1,\Lambda_2]
       
        # Training procedure parameters
        self.no_of_batches = 4000  # How many batches to go through during training.(iteration)
        self.weight_init_scaling = 3.  # 10. # default is 1. Set to larger values to get more variance in initial weights.
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

    def change_astarget(self,v): # change here the target assemblage imported from targets.py
    # def change_astarget(self,v,w):
        """set the target assemblage to a family defined in targets.py"""
        self.astarget = Bilocal_Iso_assemblage_1m_sym(v)
        # self.astarget = Bilocal_Iso_assemblage_1m_asy(v,w)
        self.asshape = tf.shape(self.astarget).numpy()

    def change_batch_size(self,cardinalities):
        self.cardinalities = cardinalities
        self.batch_size = np.prod(cardinalities) # The vector 'cardinalities' contains the cardinality of each hidden variable 
        self.batch_size_test = self.batch_size # In case we update batch_size we should also update the test batch size

def initialize():
    """ Initializes a Config class as a global variable pnn (for Parameters of Neural Network).
    The pnn object should be accessible and modifiable from all auxiliary files.
    """
    global pnn
    pnn = Config()