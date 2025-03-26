import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') #uncomment to run on CPU (faster on CPU for low dimension, may be faster at high dimensions on GPU)
import os
import matplotlib.pyplot as plt
import time
import config as cf
from utils_nn import single_run, single_evaluation_loss
from scipy.io import savemat

np.set_printoptions(10, suppress=True)

## With the same visibility
def single_sweep_training():
    """ Goes through some targets defined in config.py and return the distance to the constructed assemblage"""
    plot_v = []
    plot_Loss = []
    for H in cf.pnn.Hidden_variables:
        plot_loss = []  # Reset plot_loss for each H value

        # Choose the number of hidden variable
        cf.pnn.change_batch_size(H)
        for v in cf.pnn.visibility:
            temp_loss = []

            # choose the target family of state (change it in the config.py)
            cf.pnn.change_astarget(v)
            for trains in range(cf.pnn.nb_trains):
                print('Bilocal_Iso_assemblage_1m_sym: Starting for H = {}, v = {}, trains = {}'.format(H,v,trains + 1))

                # Run and evaluate model
                model, fit = single_run()
                print("time=", time.time() - t0)
                loss = single_evaluation_loss(model)
                temp_loss.append(loss)

                if loss < cf.pnn.threshold_val:
                    break
            if H == cf.pnn.Hidden_variables[0]:
                plot_v.append(v)
            plot_loss.append(min(temp_loss))
        plot_Loss.append(plot_loss)

    return plot_v, np.array(plot_Loss)


## With different visibilities
# def single_sweep_training():
#     """ Goes through some targets defined in config.py and return the distance to the constructed assemblage"""
#     plot_v = []
#     plot_Loss = []
#     for H in cf.pnn.Hidden_variables:
#         plot_loss = [] # Reset plot_loss for each H value
#
#         # Choose the number of hidden variable
#         cf.pnn.change_batch_size(H)
#         for v in cf.pnn.visibility:
#             for w in tuple(np.arange(0.02, v + 0.01, 0.02).round(2)):
#                 temp_loss = []
#
#                 # choose the target family of state (change it in the config.py)
#                 cf.pnn.change_astarget(v,w)
#                 for trains in range(cf.pnn.nb_trains):
#                     print('Bilocal_Iso_assemblage_1m_asy: Starting for H = {}, v = {}, w = {}, trains = {}'.format(H, v, w, trains + 1))
#
#                      # Run and evaluate model
#                     model, fit = single_run()
#                     print("time=", time.time() - t0)
#                     loss = single_evaluation_loss(model)
#                     temp_loss.append(loss)
#
#                     if loss < cf.pnn.threshold_val:
#                         break
#
#                 if H==cf.pnn.Hidden_variables[0]:
#                     plot_v.append((v,w))
#                 plot_loss.append(min(temp_loss))
#         plot_Loss.append(plot_loss)
#
#     return plot_v, np.array(plot_Loss)


if __name__ == '__main__':
    # Create directories for saving stuff
    # for dir in ['figs', 'models', 'data']:
    for dir in ['figs', 'data']:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Set up the Parameters of the Neural Network (i.e. the config object)
    t0 = time.time()
    cf.initialize()
    n = cf.pnn.dim

    # Run single sweep
    plot_v, plot_Loss = single_sweep_training()

    # Save the training data
    np.save('./data/Bilocal_Iso_assemblage_1m_v.npy', plot_v)
    np.save('./data/Bilocal_Iso_assemblage_1m_loss.npy', plot_Loss)
    # savemat('./data/Bilocal_Iso_assemblage_1m.mat', {'loss': plot_Loss, 'v': plot_v})

    # Plot the figure
    i=0
    for H in cf.pnn.Hidden_variables:
        plt.plot(plot_v, plot_Loss[i, :], label=f' H = {H}')
        i+=1
    
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('v')
    plt.savefig("./figs/Bilocal_Iso_assemblage_1m.png")
    plt.savefig("./figs/Bilocal_Iso_assemblage_1m.pdf")
    plt.show()