import numpy as np

def read_txt(path):

    with open(path, 'r') as f:
        text = f.read()

    '''
    Now, encode the text and map each character to an integer
    and vice versa. We create two dictonaries:
        1. int2char, which maps integers to characters
        2. char2int, which maps characters to unique integers
    '''
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char2int[ch] for ch in text])

    return encoded, int2char, char2int, text, chars

def get_batches(arr, n_seqs, seq_len):
    '''Create a generator that returns batches of size
       n_seqs x seq_len from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: mini-batch size, the number of sequences per batch
       seq_len: Number of sequence steps per mini-batch
    '''
    
    batch_size = n_seqs * seq_len
    n_batches = len(arr)//batch_size
    
    # removing residual characters to make full batches
    arr = arr[:n_batches * batch_size]
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], seq_len):
        x = arr[:, n:n+seq_len]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_len]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

def one_hot_encode(arr, n_labels):

    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot


'''
Our predictions come from a categorcial probability distribution
over all the possible characters.
We can make the sample text and make it more reasonable to handle
(with less variables) by only considering some  most probable characters.
This will prevent the network from giving us completely absurd characters
while allowing it to introduce some noise and randomness into the sampled text.

Typically you'll want to prime the network so you can build up a hidden state.
Otherwise the network will start out generating characters at random.
In general the first bunch of characters will be a little rough
since it hasn't built up a long history of characters to predict from.
'''
def sample(net, size, prime='The', top_k=None, cuda=False):
        
    if cuda:
        net.cuda()
    else:
        net.cpu()

    net.eval()
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = net.predict(ch, h, cuda=cuda, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = net.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)

    return ''.join(chars)
