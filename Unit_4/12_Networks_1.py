#!/usr/bin/env python
# coding: utf-8

# In[10]:


from numpy import pi, linspace, sin, diff, arange, asarray, zeros, exp, array, linspace, median, gradient, around
from numpy import zeros_like, triu_indices, triu_indices_from, tril_indices, var, mean, std, sqrt, where, isnan, nan_to_num, delete, floor
from numpy import nan, flip, argwhere, ones, diag, correlate, corrcoef, transpose, cov, flip, ceil, cos, sin, arctan
from numpy import angle, exp, amax, amin, absolute, meshgrid, fill_diagonal, concatenate, c_, real, argsort, tile
from numpy import empty_like, log, logical_and, copy, greater, invert, nonzero, count_nonzero, divide, repeat
from numpy import count_nonzero

from matplotlib.pyplot import xlabel, ylabel, hist, bar, yticks, legend, axis, figure, xticks, gca, show

from scipy.signal import butter, sosfilt
from scipy.stats import spearmanr, kendalltau

from matplotlib.pyplot import subplots, figure

from pandas import read_csv

import pyedflib


# # Networks with Python
# ## NetworkX
# 
# [__NetworkX__](https://networkx.org) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
# 
# To use NetworkX and visualise your networks, you can import the whole package.

# In[11]:


import networkx as nx


# In[ ]:


# Create Graphs

NetworkX has built-in function to produce basic and classic graphs. 

## Complete Graph

A complete graph is a network where all nodes are connected to every other node. The complete graph is undirected. 

nodes = 10

G = nx.complete_graph(nodes)

layout = nx.spring_layout(G)

nx.draw_networkx(G, pos=layout)


# ## Random graph
# 
# A random graph is a network where edges are added according to an edge probability to each node. The edge probability of the Erdos-Renyi graph is drawn from a normal distribution with a specified mean. 
# 

# In[13]:


nodes = 100

edge_probab = 0.1

ER = nx.erdos_renyi_graph(nodes, edge_probab)

layout = nx.spring_layout(ER)

nx.draw(ER, layout)


# _edge_probab_ controls the probability of assigning an edge. 
# 
# edge_probab = 0: no edge
# 
# edge_probab = 1: complete graph
# 

# ## Directed Random Graph
# 
# In a directed graph, the directionality is indicated by an arrowhead on the edge. 
# 

# In[14]:


nodes = 5
edge_prob = 0.2

ER_dir = nx.erdos_renyi_graph(nodes, edge_prob, directed=True, seed=111)

layout = nx.circular_layout(ER_dir)

# nx.draw()
nx.draw_networkx(ER_dir, layout,
        node_size=1000,
        node_color='r',
        with_labels=True,
        arrowsize=20)


print(ER_dir.nodes, '\n', ER_dir.edges)


# ## Watts-Strogatz graph
# 
# Flexible network algorithm to build scale-free, small world, etc networks. 
# 
# Based on: Duncan J. Watts and Steven H. Strogatz, Collective dynamics of small-world networks, Nature, 393, pp. 440–442 (1998).
# 
# NetworkX documentation:
# https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.watts_strogatz_graph.html
# 

# In[15]:


nodes = 120
neighbours = 8
alpha = 0.2

ws = nx.watts_strogatz_graph(nodes, neighbours, alpha, seed=111)

layout = nx.spiral_layout(ws)

#nx.draw(ws, layout,
#        node_size=700,
#        node_color='tomato',
#        with_labels=True
#       )

nx.draw_networkx_nodes(ws, pos=layout, alpha=1, node_size=200, node_color='cornflowerblue', node_shape='o')
nx.draw_networkx_edges(ws, pos=layout, alpha=.8, edge_color='tomato', width=1);


# ## Nodes and Edges information

# In[16]:


print(ws.nodes)
print('')
print(ws.edges)


# # Network Quantification

# ## Degree
# 
# Each node within a graph has a number of edges connected to it and this number is referred to as the node (or vertex) **degree**. 
# 
# In directed graphs (or digraphs) the degree can be split into the **in degree** which counts the number of edges pointing *into* the node and **out degree** which counts the number of edges emanating *from* the node. 
# 

# In[18]:


node = 11

print('Degree of node', node, 'is', ws.degree[node])
print('')
ws.degree


# ## Degree distribution
# 
# It is straightforward to look at the degrees of a network with only a few nodes. However, for large networks with many nodes, the degree will be an array with as many numbers as there are nodes. This requires a more convenient way to summarise this information. An often used solution is to look at the _degree distribution_. 
# 
# The degree distribution is normally presented as a histogram showing how many times a given degree was found in that network.

# In[17]:


from matplotlib.pyplot import subplots

fig, ax = subplots()

ax.plot(nx.degree_histogram(ws), '-*');
ax.set_xlabel('Node Degree')
ax.set_ylabel('Degree Count ');

show()


# In a __directed__ network the number of incoming connections is called the __in degree__ and the number of outgoing connections is called the __out degree__.
# 
# Thus, in an undirected network __in degree = out degree__.
# 
# Here is a more complex example:
# 

# In[20]:


nodes  = 100
probab = 0.02

G = nx.gnp_random_graph(nodes, probab, seed=1)

from matplotlib.pyplot import axes, axis, title

import collections

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = subplots()

ax.bar(deg, cnt, width=0.80, color="b")

title("Degree Distribution", fontsize=20)
ax.set_ylabel("Count", fontsize=16)
ax.set_xlabel("Degree", fontsize=16)
ax.set_xticks([d for d in deg])
ax.set_xticklabels(deg);

# draw graph in inset
axes([0.4, 0.4, 0.5, 0.5])

G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])

pos = nx.spring_layout(G, seed=2)

axis("off")

nx.draw_networkx_nodes(G, pos, node_size=30, node_color='r')
nx.draw_networkx_edges(G, pos);

show()


# This example plots the degree distribution, showing, for example, that 11 nodes in this network have no edges (degree = 0). You can verify that from the overlaid graph.
# 
# Note how the degree with highest probability (2) reflects the choice of edge probability of 2%.
# 

# ## Clustering coefficients
# 
# As an example of a more complex quantitative measure, we will look at the clustering coefficient.  
# Look at its formula below and consider extreme cases to understand what useful information the measure is supposed to convey. 
# 
# The clustering coefficient $C_i$ of node $i$ is calculated as:
# 
# $C_i = \frac{2\cdot n_i}{k\cdot(k-1)}$
# 
# where $n_i$ is the number of connections between nearest neighbors of node $i$; and $k$ is the number of nearest neighbors of node $i$.
# 
# The formula is derived as the number of edges between the neighbours divided by the maximally possible number of connections. The maximal number of possible connections of $k$ neighbours is $\frac{ k(k-1)}{2}$. There are $k\times k$ elements but if we leave out self-connections it becomes 
# $k\times (k-1)$. As each edge is included twice (forward and backward) division by 2 gives the number of undirected connections.
# 
# This yields some important properties: if there is no connection between any of the neighbours, then $e=0$ and $C_u = 0$. If all neighbours are maximally connected (each node connected to every other node), then 
# $e=\frac{ k(k-1)}{ 2 }$ and $C_u = 1$. 
# The clustering coefficient therefore tells the extent to which neighbours of a node are connected among themselves. This definition is valid for undirected networks with no self-connections.
# 
# You can obtain the clustering coefficients in a Python dictionary using `clustering`. To extract the clustering coefficients from a graph as a Python list:

# In[31]:


node = 0

print('CC of node', node, 'is', nx.clustering(ws, node))
print('')
cc_dict = nx.clustering(ws)

cc_list = [cc for cc in cc_dict.values()]

around(cc_list[:10], 2)


# And similar to the degree, for large networks, it can be meaningful to plot the distribution of clustering coefficients:

# In[36]:


fig, ax = subplots(figsize=(4, 2))

ax.hist(list(nx.clustering(ws).values()), color='b');
ax.set_xlabel('Clustering Coefficient')
ax.set_ylabel('Count');

show()


# Random and regular graphs have analytical distributions against which empirical distributions can be compared. 
