import infomap
import networkx as nx
import pandas as pd
import numpy as np
import netsci.models.random as nsr
import netsci.metrics.motifs as nsm
import netsci.visualization as nsv
import matplotlib.pyplot as plt
import community as community_louvain

from connectivity_graph_base import ConnectivityGraph


class MotifCommunityAnalyzer(ConnectivityGraph):

    def compute_motifs(self, algorithm):
        motifs = nsm.motifs(
            self.binary_adjacency_matrix.astype(int), algorithm=algorithm)
        print(motifs)
        nsv.bar_motifs(motifs)

    def create_graph_motifs(self):
        motifs = nsm.motifs(
            self.binary_adjacency_matrix.astype(int),
            algorithm='louzoun', participation=True)

        G_only_motif = nx.DiGraph()

        for triplet in motifs[1][3]:
            G_only_motif.add_node(triplet[0])
            G_only_motif.add_node(triplet[1])
            G_only_motif.add_node(triplet[2])
            G_only_motif.add_edge(triplet[0], triplet[1])
            G_only_motif.add_edge(triplet[2], triplet[1])

        new_labels = {}
        for node in G_only_motif.nodes:
            new_labels[node] = self.channels[node]
        nx.relabel.relabel_nodes(G_only_motif, new_labels, copy=False)

        nodes = set(self.G.nodes)
        m_nodes = set(G_only_motif)
        nodes_diff = nodes.difference(m_nodes)

        G_only_motif.add_nodes_from(nodes_diff)
        print("Density:", nx.density(G_only_motif))

        plt.figure(figsize=(12, 10))
        nx.draw_networkx(G_only_motif, pos=self.channel_locations, arrowsize=1,
                         node_color='lightcyan', edge_color='silver')

    def get_communities_infomap(self):
        im = infomap.Infomap("--two-level --directed")

        # new_labels = {}
        # for i, node in enumerate(self.G.nodes):
        #     new_labels[node] = i
        # G = nx.relabel.relabel_nodes(self.G, new_labels, copy=True)
        G = nx.DiGraph(self.binary_adjacency_matrix)

        for n in G.nodes():
            im.add_node(n)
        for e in G.edges():
            im.add_link(*e)
        im.run()
        partition = [[] for _ in range(im.numTopModules())]
        for node in im.iterTree():
            if node.isLeaf():
                partition[node.moduleIndex()].append(node.physicalId)

        communities = {}
        c = 0
        for com in partition:
            # print(com)
            for i in com:
                communities[self.channels[i]] = c
            c = c+1

        d = {}
        for key, value in communities.items():
            d.setdefault(value, []).append(key)

        for key, value in sorted(d.items()):
            print("Nodes that belong to community ", key, ":")
            for i in value:
                print(i, end=' ')
            print("\n")

        return communities

    def community_composition(self):
        G_undirected = self.G.to_undirected()
        community = community_louvain.best_partition(G_undirected)

        d = {}
        for key, value in community.items():
            d.setdefault(value, []).append(key)

        for key, value in sorted(d.items()):
            print("Nodes that belong to community ", key, ":")
            for i in value:
                print(i, end=' ')
            print("\n")

        return community

    def draw_community_graph(self, communities):
        values = [communities.get(node, 0.25) for node in self.G.nodes]
        self.draw_Graph(values)
