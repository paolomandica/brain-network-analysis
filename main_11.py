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

#########################################################################################################################################

                                                        #PDC#

########################################################################################################################################


#Open_eyes_analysis
values,channels,num_of_channels,num_of_samples,sample_freq = open_file(open_eyes_path)

p = None
G = None
density = None
connectivity_matrix = None
binary_adjacency_matrix = None
freq_alpha = 10

print('Computing PDC for open eyes data')

connectivity_matrix, binary_adjacency_matrix = connectivity(freq_alpha,values=values,p=p,channels=channels,sample_freq=sample_freq,G=G,density=density,connectivity_matrix = connectivity_matrix,
             binary_adjacency_matrix=binary_adjacency_matrix
                ,method='PDC', algorithm='yw',order=None,max_order=10,plot=True,resolution=100,threshold=0.2,
                 mode=0)


plt.imshow(binary_adjacency_matrix,cmap='Greys',interpolation='nearest')
plt.show()


#Closed_eyes_analysis


print('Computing PDC for closed eyes data')

values,channels,num_of_channels,num_of_samples,sample_freq = open_file(closed_eyes_path)

p = None
G = None
density = None
connectivity_matrix = None
binary_adjacency_matrix = None
freq_alpha = 10



connectivity_matrix, binary_adjacency_matrix = connectivity(freq_alpha,values=values,p=p,channels=channels,sample_freq=sample_freq,G=G,density=density,connectivity_matrix = connectivity_matrix,
             binary_adjacency_matrix=binary_adjacency_matrix
                ,method='PDC', algorithm='yw',order=None,max_order=10,plot=True,resolution=100,threshold=0.2,
                 mode=0)


plt.imshow(binary_adjacency_matrix,cmap='Greys',interpolation='nearest')
plt.show()
                                                       



########################################################################################################################################

                                                         #DTF#

########################################################################################################################################


#Open_eyes_analysis
values,channels,num_of_channels,num_of_samples,sample_freq = open_file(open_eyes_path)

p = None
G = None
density = None
connectivity_matrix = None
binary_adjacency_matrix = None
freq_alpha = 10

print('Computing DTF for open eyes data')


connectivity_matrix, binary_adjacency_matrix = connectivity(freq_alpha,values=values,p=p,channels=channels,sample_freq=sample_freq,G=G,density=density,connectivity_matrix = connectivity_matrix,
             binary_adjacency_matrix=binary_adjacency_matrix
                ,method='DTF', algorithm='yw',order=None,max_order=10,plot=True,resolution=100,threshold=0.2,
                 mode=0)


plt.imshow(binary_adjacency_matrix,cmap='Greys',interpolation='nearest')
plt.show()


#Closed_eyes_analysis

print('Computing DTF for closed eyes data')

values,channels,num_of_channels,num_of_samples,sample_freq = open_file(closed_eyes_path)

p = None
G = None
density = None
connectivity_matrix = None
binary_adjacency_matrix = None
freq_alpha = 10



connectivity_matrix, binary_adjacency_matrix = connectivity(freq_alpha,values=values,p=p,channels=channels,sample_freq=sample_freq,G=G,density=density,connectivity_matrix = connectivity_matrix,
             binary_adjacency_matrix=binary_adjacency_matrix
                ,method='DTF', algorithm='yw',order=None,max_order=10,plot=True,resolution=100,threshold=0.2,
                 mode=0)


plt.imshow(binary_adjacency_matrix,cmap='Greys',interpolation='nearest')
plt.show()