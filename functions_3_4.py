#import pyedflib
import numpy as np
import connectivipy as cp
import matplotlib.pyplot as plt
import mne
import pandas as pd
import networkx as nx
import infomap


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
        return connectivity_matrix, binary_adjacency_matrix, G


def draw_Graph(Adj,position,channels):
    G = nx.from_numpy_matrix(Adj, create_using=nx.DiGraph)
    ch_dict = { i : channels[i] for i in range(0, len(channels))}
    G = nx.relabel_nodes(G,ch_dict)
    pos = {position['label'][i] : (position['x'][i],position['y'][i]) for i in range(len(channels)) }
    f = plt.figure(figsize=(10, 10))
    nx.draw(G,pos, node_size=700, node_color = 'lightcyan',edge_color ='silver')
    nx.draw_networkx_labels(G, pos=pos, font_color='black')
    plt.show()

def draw_Graph_values(G,position,channels, values):
    ch_dict = { i : channels[i] for i in range(0, len(channels))}
    G = nx.relabel_nodes(G,ch_dict)
    pos = {position['label'][i] : (position['x'][i],position['y'][i]) for i in range(len(channels)) }
    f = plt.figure(figsize=(10, 10))
    nx.draw(G,pos, node_size=700, node_color = values,edge_color ='silver')
    nx.draw_networkx_labels(G, pos=pos, font_color='black')
    plt.show()
    return G


def create_grapf_motifs(motifs_louzoun):

  G_only_motif=nx.DiGraph()
  for triplet in motifs_louzoun[1][3]:
    G_only_motif.add_node(triplet[0])
    G_only_motif.add_node(triplet[1])
    G_only_motif.add_node(triplet[2])
    G_only_motif.add_edge(triplet[0],triplet[1])
    G_only_motif.add_edge(triplet[2],triplet[1])
    
  Adj_only_motif = nx.adjacency_matrix(G_only_motif)
  Adj_only_motif=Adj_only_motif.toarray()
  # set values of diagonal zero to avoid self-loops
  np.fill_diagonal(Adj_only_motif,0)
  return G_only_motif, Adj_only_motif

def get_communities_infomap(G):
  im = infomap.Infomap("--two-level --directed")
  for n in G.nodes():
    im.add_node(n)
  for e in G.edges():
    im.add_link(*e)
  im.run()
  partition = [[] for _ in range(im.numTopModules())]
  for node in im.iterTree():
      if node.isLeaf():
          partition[node.moduleIndex()].append(node.physicalId)
  communities={}
  c=0
  for com in partition:
    #print(com)
    for i in com:
      #print(i)
      communities[i]=c
    c=c+1
  return communities