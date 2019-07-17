import numpy as np

def mini_batch_setup(m, mini_batch_size):
    """
        function to determine the splits for mini-batch gradient descent
        if mini_batch_size == m, then performing batch gradient descent

        inputs:
            m: the total number of examples
            mini_batch_size: the number of examples in each mini-batch
    """
    num_complete_mini_batches = np.floor(m / mini_batch_size)
    incomp_mini_batch_size = 0
    if m % mini_batch_size != 0:
        incomp_mini_batch_size = m - mini_batch_size*num_complete_mini_batches
    return int(num_complete_mini_batches), int(incomp_mini_batch_size)
