#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:03:21 2021

@author: JosephNavelski
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import networkx as nx


########################################################
# Example: Social and Economic Networks by M. Jackson
########################################################

## transition probability matrix T
P = np.array([[0,.5,.5],
              [1,0,0],
              [0,1,0]])

G = nx.from_numpy_matrix(np.matrix(P), create_using=nx.DiGraph)

print("Betweenness")
b_demo = nx.betweenness_centrality(G, weight = 'weight')
for v in G.nodes():
    print(f"{v:2} {b_demo[v]:.3f}")

print("Degree centrality")
d_demo = nx.degree_centrality(G)
for v in G.nodes():
    print(f"{v:2} {d_demo[v]:.3f}")

print("Closeness centrality")
c_demo = nx.closeness_centrality(G)
for v in G.nodes():
    print(f"{v:2} {c_demo[v]:.3f}")

pos = nx.spring_layout(G, seed=367)  # Seed layout for reproducibility
nx.draw(G, pos, with_labels=True, font_weight='bold')
labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G,pos=pos,edge_labels=labels)
plt.show()

# Calculate the convergence of beliefs and concensus
def stationary_dis1(P,tol = 1E-16):
    prop = P
    # prop=np.random.rand(len(P),1)
    B=P.dot(prop)    
    err=np.inf
    iter=0
    while err>tol:
        iter+=1
        C=P.dot(B)
        err=np.linalg.norm(B-C,2)
        B=C
    return(B,err,iter)

# T-mat Iteration Way
out1_demo,out2_demo,out3_demo = stationary_dis1(P,.00000000000000000001)
out1_demo[0,:]

# Eigen Vector Way Iteration Way
w_demo, v_demo = np.linalg.eig(np.transpose(P))
stationary_probs_demo = np.real(v_demo[:,0])
print(stationary_probs_demo/sum(stationary_probs_demo))

_ = plt.hist(stationary_probs_demo/sum(stationary_probs_demo), bins=10)  # arguments are passed to np.histogram
plt.title("Histogram w/ 10 bins")
plt.show()


## Using the krackhardt_kite_graph
G = nx.krackhardt_kite_graph()

print("Betweenness")
b_demo = nx.betweenness_centrality(G)
for v in G.nodes():
    print(f"{v:2} {b_demo[v]:.3f}")

print("Degree centrality")
d_demo = nx.degree_centrality(G)
for v in G.nodes():
    print(f"{v:2} {d_demo[v]:.3f}")

print("Closeness centrality")
c_demo = nx.closeness_centrality(G)
for v in G.nodes():
    print(f"{v:2} {c_demo[v]:.3f}")

pos = nx.spring_layout(G, seed=367)  # Seed layout for reproducibility
nx.draw(G, pos)
plt.show()


# Betweenness centrality measures of positive gene functional 
# associations using WormNet v.3-GS.
## Exotic Betweeness Centrality Graph

from random import sample
import networkx as nx
import matplotlib.pyplot as plt

# Gold standard data of positive gene functional associations
# from https://www.inetbio.org/wormnet/downloadnetwork.php
G = nx.read_edgelist(r"Desktop/Soc_and_Econ_Networks/WormNet.v3.benchmark.txt")

# remove randomly selected nodes (to make example fast)
num_to_remove = int(len(G) / 1.5)
nodes = sample(list(G.nodes), num_to_remove)
G.remove_nodes_from(nodes)

# remove low-degree nodes
low_degree = [n for n, d in G.degree() if d < 10]
G.remove_nodes_from(low_degree)

# largest connected component
components = nx.connected_components(G)
largest_component = max(components, key=len)
H = G.subgraph(largest_component)

# compute centrality
centrality = nx.betweenness_centrality(H, k=10, endpoints=True)

# compute community structure
lpc = nx.community.label_propagation_communities(H)
community_index = {n: i for i, com in enumerate(lpc) for n in com}

#### draw graph ####
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(H, k=0.15, seed=4572321)
node_color = [community_index[n] for n in H]
node_size = [v * 20000 for v in centrality.values()]
nx.draw_networkx(
    H,
    pos=pos,
    with_labels=False,
    node_color=node_color,
    node_size=node_size,
    edge_color="gainsboro",
    alpha=0.4,
)

# Title/legend
font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title("Gene functional association network (C. elegans)", font)
# Change font color for legend
font["color"] = "r"

ax.text(
    0.80,
    0.10,
    "node color = community structure",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = betweeness centrality",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)

# Resize figure for label readibility
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.show()

