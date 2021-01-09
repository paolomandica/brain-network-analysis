import pyedflib
import numpy as np
import connectivipy as cp
import matplotlib.pyplot as plt
import mne
import networkx as nx
import pandas as pd
from functions import *



#data paths
open_eyes_path = 'data/S002/S002R01.edf'
closed_eyes_path = 'data/S002/S002R02.edf'
location_path = 'data/channel_locations.txt'

#Plot graph for open eyes PDC 

position=pd.read_csv(location_path, sep='\s+')

values_open,channels_open,num_of_channels_open,num_of_samples_open,sample_freq_open = open_file(open_eyes_path)

connectivity_matrix_open, binary_adjacency_matrix_open = connectivity(freq=10,values=values_open,p=None,channels=channels_open,
    sample_freq=sample_freq_open,G=None,density=None,connectivity_matrix = None,
             binary_adjacency_matrix=None
                ,method='PDC', threshold=0.05)


position['label']=channels_open

draw_Graph(binary_adjacency_matrix_open,position=position,channels=channels_open)


#Plot graph for closed eyes PDC

values_closed,channels_closed,num_of_channels_closed,num_of_samples_closed,sample_freq_closed = open_file(closed_eyes_path)

connectivity_matrix_closed, binary_adjacency_matrix_closed = connectivity(freq=10,values=values_closed,p=None,channels=channels_closed,
    sample_freq=sample_freq_closed,G=None,density=None,connectivity_matrix = None,
            binary_adjacency_matrix=None
            ,method='PDC', threshold=0.05)

draw_Graph(binary_adjacency_matrix_closed,position=position,channels=channels_closed)

