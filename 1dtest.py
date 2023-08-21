import numpy as np
import matplotlib.pyplot as plt

class Source:
    def __init__(self, x, y, gamma1, gamma2, f1, f2):
        self.x = x
        self.y = y
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.f1 = f1
        self.f2 = f2
        
    def calc_shear(self, lenses):
        