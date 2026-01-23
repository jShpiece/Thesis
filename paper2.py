import numpy as np
import matplotlib.pyplot as plt

import arch.pipeline as pipeline
import arch.source_obj as source_obj
import arch.utils as utils
import arch.main as main

class NFWE_Halo:
    def __init__(self, mass, concentration, orientation, redshift):
        self.mass = mass
        self.concentration = concentration
        self.redshift = redshift
        # Additional initialization code here

def main():
    # Create a halo object
    halo = NFWE_Halo(mass=1e14, concentration=5, orientation=0, redshift=0.3)

    # Set up the source object
    