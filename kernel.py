import numpy as np


class Kernel(object):
    """Check kernels here https://en.wikipedia.org/wiki/Support_vector_machine"""
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

 
    @staticmethod
    def gaussian(sigma):
        if (sigma <= 0):
            raise ValueError('A value of sigma can not be smaller or equals to zero')
        return lambda x, y: np.exp(-1 * np.linalg.norm(x-y)**2 / 2* sigma ** 2)
    
    @staticmethod
    def polynomial(d, typeKernel):
        if (d <= 0):
            raise ValueError('A value of d can not be smaller or equals to zero')
        return lambda x, y: (1 + x.T*y)**d