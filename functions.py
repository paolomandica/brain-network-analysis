import pyedflib
import numpy as np
import connectivipy as cp
import matplotlib.pyplot as plt
import mne
import pandas as pd
import networkx as nx

chan_19 = 'Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2'.split(' ')


def open_file(path,sub_channels=False):
    open_eyes_raw = mne.io.read_raw_edf(path)
    open_eyes_df = open_eyes_raw.to_data_frame()
    sample_freq = open_eyes_raw.info['sfreq']
    open_eyes_data = open_eyes_df.drop(['time'],axis=1)
    if sub_channels:
        channels = list(map(lambda x:x.strip('.'),open_eyes_data.columns))
        open_eyes_data.columns = channels
        channels = 'Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2'.split(' ')
        open_eyes_data = open_eyes_data[channels]
        values = open_eyes_data.T.values
        num_of_channels,num_of_samples = values.shape
    else:
        values = open_eyes_data.T.values
        channels = list(map(lambda x:x.strip('.'),open_eyes_data.columns))
        num_of_channels, num_of_samples = values.shape
    return values,channels,num_of_channels,num_of_samples,sample_freq

def connectivity(freq,values,p,channels,sample_freq,G,density,connectivity_matrix,binary_adjacency_matrix
                , method,algorithm='yw',order=None,max_order=10,plot=False,resolution=100,threshold=None):

        if not order:
            best,crit = cp.Mvar.order_akaike(values,max_order) 
            if plot:
                plt.plot(1+np.arange(len(crit)), crit,marker='o', linestyle='dashed',markersize=8,markerfacecolor='yellow')
                plt.grid()
                plt.show()
            p = best
            print()
            print('best model order p: {}'.format(best))
            print()
        else:
            p = order
        data = cp.Data(values,chan_names=channels)
        data.fit_mvar(p, algorithm)
        #multivariate model coefficient (see slides)
        ar, vr = data.mvar_coefficients
        if method == 'DTF':
            Adj=cp.conn.dtf_fun(ar,vr,fs = sample_freq, resolution = 100)[freq,:,:]
        else:
            Adj=cp.conn.pdc_fun(ar,vr,fs = sample_freq, resolution = 100)[freq,:,:]
            
        np.fill_diagonal(Adj,0)
        
        #create Graph from Adj matrix
        G = nx.from_numpy_matrix(np.array(Adj), create_using=nx.DiGraph)

        A = nx.adjacency_matrix(G)
        
        A=A.toarray()

        # set values of diagonal zero to avoid self-loops
        np.fill_diagonal(A,0)
        
        #reduce Graph density
        while(nx.density(G)>threshold):
            #find min values different from zeros
            arg_min = np.argwhere(A == np.min(A[np.nonzero(A)]))
            i,j = arg_min[0][0],arg_min[0][1]
            #remove i,j edge from the graph
            G.remove_edge(i,j)
            #recalculate the graph
            A = nx.adjacency_matrix(G)
            A=A.toarray()
            np.fill_diagonal(A,0)
            #np.fill_diagonal(A,diag)
      
        density = nx.density(G)
        connectivity_matrix = A.copy()
        A[A>0] = 1
        binary_adjacency_matrix = A
        return connectivity_matrix, binary_adjacency_matrix


def draw_Graph(Adj,position,channels):
    G = nx.from_numpy_matrix(Adj, create_using=nx.DiGraph)
    ch_dict = { i : channels[i] for i in range(0, len(channels))}
    G = nx.relabel_nodes(G,ch_dict)
    pos = {position['label'][i] : (position['x'][i],position['y'][i]) for i in range(len(channels)) }
    f = plt.figure(figsize=(10, 10))
    nx.draw(G,pos, node_size=700, node_color = 'lightcyan',edge_color ='silver')
    nx.draw_networkx_labels(G, pos=pos, font_color='black')
    plt.show()


def significance(values,method,max_order,order=None,signf_threshold=0.05,channels=chan_19,Nrep=200,alpha=0.05):
        if not order:
            best,crit = cp.Mvar.order_akaike(values,max_order)
            plt.plot(1+np.arange(len(crit)), crit,marker='o', linestyle='dashed',markersize=8,markerfacecolor='yellow')
            plt.grid()
            plt.show()
        else:
            p=order
        p = best
        data = cp.Data(values,chan_names=channels)
        data.fit_mvar(p,'yw')
        if method == 'DTF':
            matrix_values = data.conn('dtf')
        else:
            matrix_values = data.conn('pdc')
        significance_matrix = data.significance(Nrep=Nrep, alpha=alpha,verbose=False)
        significance_matrix[significance_matrix<0.05] = 1
        significance_matrix[significance_matrix!=1] = 0
        return significance_matrix