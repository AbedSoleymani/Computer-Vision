import numpy as np

def fft(input):
    '''This function takes in a normalized, grayscale image
       and returns a frequency spectrum transform of that image. '''
    f = np.fft.fft2(input)
    fshift = np.fft.fftshift(f)
    spectrum = 20*np.log(np.abs(fshift))
    
    return spectrum