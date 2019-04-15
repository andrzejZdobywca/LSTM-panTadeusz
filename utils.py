import numpy as np
import torch 

def read_text(path):
    ''' open file and return it as a string

        Arguments:
        path: string represeting PATH to a file
    '''

    with open(path, 'r') as file:
        data = file.read()
    return data

def convert_to_int(data):
    ''' Convert a string to an array of integers, 
        which are uniquely mapped to a character

        Arguments:
        data: String that is to be converted
    '''

    chars = set(data)
    chars_as_int = dict([(x, i) for i, x in enumerate(chars)])
    data_as_int = np.array([chars_as_int[x] for x in data])
    return data_as_int

def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''
    
    batch_size_total = batch_size * seq_length

    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]

    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        x, y = torch.Tensor(x).long(), torch.Tensor(y).long()
        yield x, y