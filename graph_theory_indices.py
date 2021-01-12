import pyedflib
import numpy as np
import connectivipy as cp
import matplotlib.pyplot as plt
import mne
import networkx as nx
import bct

from connectivity_graph_base import ConnectivityGraph
from easydict import EasyDict as edict
from time import time
from networkx.algorithms import average_shortest_path_length, average_clustering


class GraphTheoryIndices(ConnectivityGraph):
    """Class extending ConnectivityGraph with methods to analyze
    Graph Theory Indices (ex. 2)."""

    def compute_global_indices(self, weighted=False):
        """Computes Average Clustering Coefficient and Average Path Length
        of the graph (binary or weighted).

        Args:
        ----------
        weighted : boolean (default=False)
            If True, use the weighted graph, otherwise use the binary one.

        Returns:
        ----------
        avg_cl_coef : float
            Average clustering coefficient
        avg_path_len : float
            Average path length
        """
        if weighted:
            avg_cl_coef = average_clustering(self.Gw, weight='weight')
            avg_path_len = average_shortest_path_length(self.Gw,
                                                        weight='weight')
        else:
            avg_cl_coef = average_clustering(self.G)
            avg_path_len = average_shortest_path_length(self.G)
        return avg_cl_coef, avg_path_len

    def compute_local_indices(self, weighted=False):
        """Computes degree, in-degree and out-degree of the graph
        (binary or weighted).

        Args:
        ----------
        weighted : boolean (default=False)
            If True, use the weighted graph, otherwise use the binary one.

        Returns:
        ----------
        degree : list of 2-tuples (node, degree)

        in_degree : list of 2-tuples (node, in_degree)

        out_degree : list of 2-tuples (node, out_degree)
        """
        if weighted:
            G = self.Gw
        else:
            G = self.G

        degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        in_degree = sorted(
            G.in_degree(), key=lambda x: x[1], reverse=True)
        out_degree = sorted(
            G.out_degree(), key=lambda x: x[1], reverse=True)
        return degree, in_degree, out_degree

    def plot_global_indices(self, thresholds):
        """Computes and plots the variation of global indices (ACC, APL)
        of graphs constructed with different density thresholds.

        Args:
        -----------
        thresholds : list of floats
            List of density thresholds used to build the graphs.
        """
        avg_cl_coeffs = []
        avg_path_lens = []

        start = time()
        for t in thresholds:
            print("Computing for t =", t)
            self.compute_connectivity(freq=10, threshold=t)
            try:
                acc, apl = self.compute_global_indices()
                avg_cl_coeffs.append(acc)
                avg_path_lens.append(apl)
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
        """Compute the small-worldness index of the binary directed graph.

        Returns:
        -----------
        SMI : float
        """
        randMetrics = {"C": [], "L": []}
        print("Computing random graphs...")
        for i in range(10):
            rand_adj = bct.randmio_dir(self.binary_adjacency_matrix, 10, i)[0]
            G_r = nx.convert_matrix.from_numpy_array(
                rand_adj, create_using=nx.DiGraph)
            randMetrics["C"].append(average_clustering(G_r))
            randMetrics["L"].append(average_shortest_path_length(G_r))

        print("Computing SMI...")
        C, L = self.compute_global_indices()
        Cr = np.mean(randMetrics["C"])
        Lr = np.mean(randMetrics["L"])

        SMI = (C / Cr) / (L / Lr)
        print("Completed!")
        return SMI
