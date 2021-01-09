import pyedflib
import numpy as np
import connectivipy as cp
import matplotlib.pyplot as plt
import mne
import networkx as nx
import bct

from easydict import EasyDict as edict
from time import time
from networkx.algorithms import average_shortest_path_length, average_clustering


class GraphTheoryIndices:
    def __init__(self, path):
        self.path = path
        self.G = nx.DiGraph()
        self.sample_freq = None
        self.values = None
        self.channels = None
        self.num_of_channels = None
        self.num_of_samples = None
        self.read_edf_data()

    def read_edf_data(self):
        raw = mne.io.read_raw_edf(self.path)
        df = raw.to_data_frame()
        self.sample_freq = raw.info['sfreq']
        df = df.drop(['time'], axis=1)
        self.values = df.T.values
        self.channels = list(map(lambda x: x.strip('.'), df.columns))
        self.num_of_channels, self.num_of_samples = self.values.shape
        print("EDF data loaded!")

    def compute_connectivity(self, freq, method="PDC", algorithm="yw",
                             order=None, max_order=10, plot=False,
                             resolution=100, threshold=None):

        if not order:
            best, crit = cp.Mvar.order_akaike(self.values, max_order)
            if plot:
                plt.plot(1+np.arange(len(crit)), crit, marker='o',
                         linestyle='dashed', markersize=8, markerfacecolor='yellow')
                plt.grid()
                plt.show()
            p = best
        else:
            p = order

        data = cp.Data(self.values, chan_names=self.channels)
        data.fit_mvar(p, algorithm)
        # multivariate model coefficient (see slides)
        ar, vr = data.mvar_coefficients
        if method == 'DTF':
            Adj = cp.conn.dtf_fun(ar, vr, fs=self.sample_freq,
                                  resolution=100)[freq, :, :]
        else:
            Adj = cp.conn.pdc_fun(ar, vr, fs=self.sample_freq,
                                  resolution=100)[freq, :, :]

        np.fill_diagonal(Adj, 0)

        # create Graph from Adj matrix
        G = nx.from_numpy_matrix(np.array(Adj), create_using=nx.DiGraph)
        A = nx.adjacency_matrix(G)
        A = A.toarray()

        # set values of diagonal zero to avoid self-loops
        np.fill_diagonal(A, 0)

        # reduce Graph density
        while(nx.density(G) > threshold):
            # find min values different from zeros
            arg_min = np.argwhere(A == np.min(A[np.nonzero(A)]))
            i, j = arg_min[0][0], arg_min[0][1]
            # remove i,j edge from the graph
            G.remove_edge(i, j)
            # recalculate the graph
            A = nx.adjacency_matrix(G)
            A = A.toarray()
            np.fill_diagonal(A, 0)
            # np.fill_diagonal(A,diag)

        density = nx.density(G)
        connectivity_matrix = A.copy()
        A[A > 0] = 1
        binary_adjacency_matrix = A

        self.connectivity_matrix = connectivity_matrix
        self.binary_adjacency_matrix = binary_adjacency_matrix

        G = nx.DiGraph(binary_adjacency_matrix)
        new_labels = {}
        for i, node in enumerate(G.nodes):
            new_labels[node] = self.channels[i]
        self.G = nx.relabel.relabel_nodes(G, new_labels, copy=True)

    def compute_global_indices(self):
        self.avg_cl_coef = average_clustering(self.G)
        self.avg_path_len = average_shortest_path_length(self.G)

    def compute_local_indices(self):
        self.degree = sorted(self.G.degree(), key=lambda x: x[1], reverse=True)
        self.in_degree = sorted(
            self.G.in_degree(), key=lambda x: x[1], reverse=True)
        self.out_degree = sorted(
            self.G.out_degree(), key=lambda x: x[1], reverse=True)

    def plot_global_indices(self, thresholds):
        avg_cl_coeffs = []
        avg_path_lens = []

        start = time()
        for t in thresholds:
            print("Computing for t =", t)
            self.compute_connectivity(freq=10, threshold=t)
            try:
                self.compute_global_indices()
                avg_cl_coeffs.append(self.avg_cl_coef)
                avg_path_lens.append(self.avg_path_len)
            except:
                avg_cl_coeffs.append(0)
                avg_path_lens.append(0)
        seconds = int(time() - start)
        print("Time passed: %d seconds" % (seconds))

        plt.figure(figsize=(6, 4))
        plt.title("Avg clustering coeff behavior")
        plt.plot(thresholds, avg_cl_coeffs)
        plt.xlabel("Threshold")
        plt.ylabel("Avg clustering coeff")
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.title("Avg path length behavior")
        plt.plot(thresholds, avg_path_lens)
        plt.xlabel("Threshold")
        plt.ylabel("Avg path length")
        plt.show()

    def compute_SMI(self):
        # Compute the mean clustering coefficient and average shortest path length
        # for an equivalent random graph
        randMetrics = {"C": [], "L": []}
        print("Computing random graphs...")
        for i in range(10):
            rand_adj = bct.randmio_dir(self.binary_adjacency_matrix, 10, i)[0]
            G_r = nx.convert_matrix.from_numpy_array(
                rand_adj, create_using=nx.DiGraph)
            randMetrics["C"].append(average_clustering(G_r))
            randMetrics["L"].append(average_shortest_path_length(G_r))

        print("Computing SMI...")
        self.compute_global_indices()
        C = self.avg_cl_coef
        L = self.avg_path_len
        Cr = np.mean(randMetrics["C"])
        Lr = np.mean(randMetrics["L"])

        self.SMI = (C / Cr) / (L / Lr)
        print("Completed!")
