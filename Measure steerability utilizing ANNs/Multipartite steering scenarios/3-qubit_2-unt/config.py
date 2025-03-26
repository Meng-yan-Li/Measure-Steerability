import numpy as np
import tensorflow as tf 
from targets import GHZ_assemblage_2unt_2m,GHZ_assemblage_2unt_3m

class Config:
    """ Config is just a collection of all metadata which can be accessed from auxiliary files as well. """
    def __init__(self):
        self.dim = 2 # Dimensionality of Hilbert-space.
        self.oa = 2 # Number of outcomes of Alice.
        self.ob = 2 # Number of outcomes of Bob.
        self.ma = 2 # Number of measurements of Alice.
        self.mb = 2 # Number of measurements of Bob.
        self.settings = np.array([[self.oa,self.ma],[self.ob,self.mb]]) # Settings of untrusted party: [[oa,ma],[ob,mb],...]
        self.Det_strategy = deterministic_strategy(self.settings) # Det_strategy[i] represent the strategy of (i+1)th party.
        # Visibility parameter of target assemblage
        self.visibility = (0, .1, .2, .3, .4, .5, .6, .66, 2/3, .67, .7, .8, .9, 1) # GHZ_assemblage_2unt_2m
        # self.visibility = (0, .1, .2, .3, .4, .42, .4285, .43, .5, .6, .7, .8, .9, 1) # GHZ_assemblage_2unt_3m

        # Neural network parameters
        self.inputsize= 2**2*2**2 # oa^{ma}*ob^{mb}, i.e., the cardinality of the hidden variable.
        self.lambda_outputsize = 2*self.dim+1 # variables for \rho_\lambda_C, and one for weight distribution.
        self.latin_depth = 1
        self.latin_width = 200
        self.nb_trains = 1  # choose how many times to train each target (numeric artifact that could be eliminated by increasing the number of trials per assemblage.)
        self.threshold_val = 0.001 # Threshold for stopping training.
        self.Hidden_variables = (self.inputsize,) # The cardinality of \lambda

        # Training procedure parameters
        self.no_of_batches = 4000  # How many batches to go through during training.(iteration)
        self.weight_init_scaling = 2.  # Default is 1. Set to larger values to get more variance in initial weights. Increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
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

    def change_astarget(self, v): # Change here the target assemblage imported from targets.py
        """set the target assemblage to a family defined in targets.py"""
        self.astarget = GHZ_assemblage_2unt_2m(v)
        self.asshape = tf.shape(self.astarget).numpy()

    def change_batch_size(self,new_batch_size):
        self.batch_size = new_batch_size 
        self.batch_size_test = self.batch_size # In case we update batch_size we should also update the test batch size

def deterministic_strategy(arr):
    parties = arr.shape[0] # Each party may have different number of inputs and outputs
    Det_strategy = []
    for p in range(parties):
        oa = arr[p,0]  # Number of outcomes
        ma = arr[p,1]  # Number of measurements
        Ndet = oa ** ma  # Number of deterministic behaviors
        SingleParty = np.zeros((oa, ma, Ndet))  # Initialize array
        for lam in range(Ndet):
            # Generate the string of outcomes a (for each x) for the given variable lam
            lamdec = np.array(list(np.base_repr(lam, base=oa).zfill(ma)), dtype=int)
            
            for x in range(ma):
                for a in range(oa):
                    SingleParty[a, x, lam] = int(lamdec[x] == a)
                    # Probability is 1 if a == lamdec[x], 0 otherwise
        Det_strategy.append(SingleParty)
    # Det_strategy[i] represent the strategy of (i+1)th party
    return Det_strategy

def initialize():
    """ Initializes a Config class as a global variable pnn (for Parameters of Neural Network).
    The pnn object should be accessible and modifiable from all auxiliary files.
    """
    global pnn
    pnn = Config()