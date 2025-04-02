import networkx as nx
from pyvis.network import Network

# Load the GraphML file
G = nx.read_graphml('.\dickens\graph_chunk_entity_relation.graphml')

# Create a Pyvis network

net = Network(notebook=True, cdn_resources='in_line')  # or 'remote'

# Convert NetworkX graph to Pyvis network
net.from_nx(G)

net.show("knowledge_graph.html")