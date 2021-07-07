import numpy as np
import itertools

# Frequently used commands
inv, ax, det = np.linalg.inv, np.newaxis, np.linalg.det
cos, pi, arccos, log = np.cos, np.pi, np.arccos, np.log
sin, arcsin, exp = np.sin, np.arcsin, np.exp


class ValueFunctionIterator:

    def __init__(self, initializer_dict,
                 ):

        self.price = np.zeros(initializer_dict['price_support_size'])
