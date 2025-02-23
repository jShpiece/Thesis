# I want to use this script to read the flexion data and make sure I understand i
# 

flex_path = 'JWST_Data/JWST/ABELL_2744/Catalogs/multiband_flexion.pkl'
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

with open(flex_path, 'rb') as f:
    flex = pickle.load(f)

print(flex.keys())

# Eliminate any row with NaN values
flex = flex.dropna()
print(flex.head())