import pyedflib
import numpy as np
import connectivipy as cp
import matplotlib.pyplot as plt
import mne
import networkx as nx
from functions_11 import *


#data paths
open_eyes_path = 'data/S002/S002R01.edf'
closed_eyes_path = 'data/S002/S002R02.edf'

#open eyes variables
values_open,channels_open,num_of_channels_open,num_of_samples_open,sample_freq_open = open_file(open_eyes_path)

#closed eyes variables
values_closed,channels_closed,num_of_channels_closed,num_of_samples_closed,sample_freq_closed = open_file(closed_eyes_path)


m_open = [] 
m_closed = []
densities = [0.01,0.05,0.1,0.2,0.3,0.5]
for d in densities:
    #open PDC
    connectivity_matrix_open, binary_adjacency_matrix_open = connectivity(freq=10,values=values_open,p=None,channels=channels_open,
    sample_freq=sample_freq_open,G=None,density=None,connectivity_matrix = None,
             binary_adjacency_matrix=None
                ,method='PDC', threshold=d)
    m_open.append(binary_adjacency_matrix_open)
    #closed PDC
    connectivity_matrix_closed, binary_adjacency_matrix_closed = connectivity(freq=10,values=values_closed,p=None,channels=channels_closed,
    sample_freq=sample_freq_closed,G=None,density=None,connectivity_matrix = None,
             binary_adjacency_matrix=None
                ,method='PDC', threshold=d)
    m_closed.append(binary_adjacency_matrix_closed)
    
m = [m_open,m_closed]


#Final plot of al the cases

cols = ['Case: {}'.format(col) for col in ['open','closed']]
rows = ['Density: {}'.format(row) for row in densities]

fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(20,12))

for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=90, size='large')
    
r = 0
for row in axes:
    c = 0
    for col in row:
        col.imshow(m[c][r],cmap='Greys',interpolation='nearest')
        c += 1
    r += 1    

fig.tight_layout()
fig.subplots_adjust(left=0.05, top=0.95)
plt.show()