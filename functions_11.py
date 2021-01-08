import pyedflib
import numpy as np
import connectivipy as cp
import matplotlib.pyplot as plt
import mne
import networkx as nx




def open_file(path):
    open_eyes_raw = mne.io.read_raw_edf(path)
    open_eyes_df = open_eyes_raw.to_data_frame()
    sample_freq = open_eyes_raw.info['sfreq']
    open_eyes_data = open_eyes_df.drop(['time'],axis=1)
    values = open_eyes_data.T.values
    channels = list(map(lambda x:x.strip('.'),open_eyes_data.columns))
    num_of_channels, num_of_samples = values.shape
    return values,channels,num_of_channels,num_of_samples,sample_freq

def connectivity(freq,values,p,channels,sample_freq,G,density,connectivity_matrix,binary_adjacency_matrix
                , method,algorithm='yw',order=None,max_order=10,plot=False,resolution=100,threshold=None,
                 mode=0):

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