import pyedflib
import numpy as np
import connectivipy as cp
import matplotlib.pyplot as plt
import mne
import networkx as nx


class ConnectivityGraph:
    """Class to handle data and methods for the creation of the
    Connectivity Graphs, directed binary and directed weighted.
    """

    def __init__(self, path, sub_channels=False):
        """
        Args:
        ----------
        path : string
            Filepath of the EDF file.
        """
        self.channel_loc_path = "./data/channel_locations.txt"
        self.sample_freq = None
        self.values = None
        self.channels = None
        self.num_of_channels = None
        self.num_of_samples = None
        self.read_edf_data(path, sub_channels)
        self.channel_locations = None
        self.connectivity_matrix = None
        self.binary_adjacency_matrix = None
        self.G = None
        self.Gw = None

    def read_edf_data(self, path, sub_channels=False):
        """Reads the EDF file and saves data and info as attributes
        of the class instance.

        Args:
        ----------
        path : string
            Filepath of the EDF file.
        """
        raw = mne.io.read_raw_edf(path)
        df = raw.to_data_frame()
        self.sample_freq = raw.info['sfreq']
        df = df.drop(['time'], axis=1)
        self.channels = list(map(lambda x: x.strip('.'), df.columns))
        df.columns = self.channels

        if sub_channels:
            self.channels = 'Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2'.split(
                ' ')
            df = df[self.channels]

        self.values = df.T.values
        self.num_of_channels, self.num_of_samples = self.values.shape
        print("EDF data loaded!")

    def load_channel_locations(self):
        locations = {}

        with open(self.channel_loc_path, newline='') as fp:
            _ = fp.__next__()

            for line in fp:
                _, label, x, y = line.split()
                label = label.rstrip(".")
                x = float(x)
                y = float(y)
                locations[label] = (x, y)

            self.channel_locations = locations

    def compute_connectivity(self, freq, method="PDC",
                             order=None, max_order=10, plot=False,
                             resolution=100, threshold=None):
        """Compute the connectivity matrix and the binary matrix of the EEG data,
        using PDC or DTF method for the estimation.

        Args:
        -----------
        freq : int
            Frequency value for the connectivity matrix.
        method : string
            "PDC" or "EDF". Estimation method for connectivity.
        order : int
            Value of order of autoregressive multivariate model.
        max_order : int
            Max order computable by akaike algorithm.
        plot : boolean
            Whether to plot the akaike algo results or not.
        resolution : int
            Number of spectrum datapoints.
        threshold : float
            Density threshold for the computation of the connectivity matrix.
            Between 0 and 1.
        """
        if not order:
            best, crit = cp.Mvar.order_akaike(self.values, max_order)
            p = best
            if plot:
                plt.plot(1+np.arange(len(crit)), crit, marker='o',
                         linestyle='dashed', markersize=8, markerfacecolor='yellow')
                plt.grid()
                plt.show()
                print('Best model order p: {}'.format(best))
        else:
            p = order

        data = cp.Data(self.values, chan_names=self.channels)
        data.fit_mvar(p, "ym")
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

        # load coordinates
        self.load_channel_locations()

        # create directed binary graph
        G = nx.DiGraph(binary_adjacency_matrix)
        new_labels = {}
        for i, node in enumerate(G.nodes):
            new_labels[node] = self.channels[i]
        self.G = nx.relabel.relabel_nodes(G, new_labels, copy=True)
        # nx.set_node_attributes(self.G, self.channel_locations, "pos")

        # create directed weighted graph
        Gw = nx.DiGraph(connectivity_matrix)
        self.Gw = nx.relabel.relabel_nodes(Gw, new_labels, copy=True)
        # nx.set_node_attributes(self.Gw, self.channel_locations, "pos")

    def significance(self, method="DTF", max_order, order=None,
                     signf_threshold=0.05, Nrep=200, alpha=0.05):
        """Compute and plot the binary matrix having as positive elements
        the related p-values of the significance matrix less than threshold.

        Args:
        -----------
        method : string
            "PDC" or "EDF". Estimation method for connectivity.
        order : int
            Value of order of autoregressive multivariate model.
        max_order : int
            Max order computable by akaike algorithm.
        signf_threshold : float
            Threshold for p-values of the significance matrix.
        Nrep : int
            Number of resamples for the computation of the significance matrix.
        alpha : float
            Type 1 error rate for the significance level.
        """
        if not order:
            best, crit = cp.Mvar.order_akaike(self.values, max_order)
            plt.plot(1+np.arange(len(crit)), crit, marker='o',
                     linestyle='dashed', markersize=8, markerfacecolor='yellow')
            plt.grid()
            plt.show()
            p = best
        else:
            p = order
        print('Best model order p: {}'.format(p))

        data = cp.Data(self.values, chan_names=self.channels)
        data.fit_mvar(p, 'yw')
        if method == 'DTF':
            matrix_values = data.conn('dtf')
        else:
            matrix_values = data.conn('pdc')
        significance_matrix = data.significance(
            Nrep=Nrep, alpha=alpha, verbose=False)
        significance_matrix[significance_matrix < 0.05] = 1
        significance_matrix[significance_matrix != 1] = 0
        self.significance_matrix = significance_matrix
        plt.imshow(significance_matrix, cmap='Greys', interpolation='nearest')
        plt.show()

    def draw_Graph(self, values=None):
        if values is not None:
            node_color = values
        else:
            node_color = 'lightcyan'

        plt.figure(figsize=(12, 10))
        nx.draw_networkx(self.G, pos=self.channel_locations, arrowsize=1,
                         node_color=node_color, edge_color='silver')
