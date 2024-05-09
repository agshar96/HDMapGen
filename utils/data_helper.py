###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import torch
import pickle
import numpy as np
import networkx as nx
from HDMapGen.nuplanprocess import get_data_nuplan

__all__ = [
    'save_graph_list', 'create_graphs'
]

# save a list of graphs
def save_graph_list(G_list, fname):
  with open(fname, "wb") as f:
    pickle.dump(G_list, f)

def create_grid_with_embed(x_nodes, y_nodes, side_x = None, side_y = None, normalized = True):
    '''
    This function creates a grid with x_nodes in x-direction and y_nodes
    in y-direction. Each node has a 2D coordinate associated with it.
    The side-x and side-y specify side length in x and direction respectively.

    x_nodes: Number of nodes in x-direction
    y_nodes: Number of nodes in y-direction
    side_x: Sidelength in x-direction
    side_y: Sidelength in y-direction
    normalized: are nodes normalized or not
    '''

    if side_x is None and side_y is not None:
        side_x = side_y
    elif side_x is not None and side_y is None:
        side_y = side_x
    elif side_x is None and side_y is None:
        side_x = side_y = 1
    
    mean_tensor = torch.tensor([0,0])
    std_tensor = torch.tensor([1,1])

    if normalized:
      '''
        Getting mean and std of all nodes
      '''
      coord_list = []
      for i in range(x_nodes):
        for j in range(y_nodes):
            coords = torch.tensor([float(i*side_x), float(j*side_y)])
            coord_list.append(coords)
      coord_list = torch.stack(coord_list)
      mean_tensor = torch.mean(coord_list, dim=0)
      std_tensor = torch.var(coord_list, dim = 0)
    
    G = nx.Graph()
    num_node = 0
    for i in range(x_nodes):
        for j in range(y_nodes):
            coords = torch.tensor([float(i*side_x), float(j*side_y)]) - mean_tensor
            coords = coords / std_tensor
            G.add_node(num_node, features = coords)

            if j > 0:
                G.add_edge(num_node, num_node-1)
                G.add_edge(num_node -1, num_node)
            if i > 0:
                G.add_edge(num_node, num_node - y_nodes)
                G.add_edge(num_node - y_nodes, num_node)
            num_node += 1
    return G

def create_subnode_with_embed(x_nodes, y_nodes, side_x = None, side_y = None, subdivisions=10,
                              normalized = False):
  '''
    This function creates a grid with 'x_nodes' in x-direction and 'y_nodes'
    in y-direction. Each node has a 2D coordinate associated with it.
    The side-x and side-y specify side length in x and direction respectively.
    Each edge is further subdivided into 'subdivisions' amount of sub-nodes

    x_nodes: Number of nodes in x-direction
    y_nodes: Number of nodes in y-direction
    side_x: Sidelength in x-direction
    side_y: Sidelength in y-direction
    subdivisions: Number of subdivisions of each edge 
  '''

  if side_x is None and side_y is not None:
        side_x = side_y
  elif side_x is not None and side_y is None:
      side_y = side_x
  elif side_x is None and side_y is None:
      side_x = side_y = 1
  
  mean_tensor = torch.tensor([0,0])
  std_tensor = torch.tensor([1,1])

  if normalized:
    '''
      Getting mean and std of all nodes
    '''
    coord_list = []
    for i in range(x_nodes):
      for j in range(y_nodes):
          coords = torch.tensor([float(i*side_x), float(j*side_y)])
          coord_list.append(coords)
    coord_list = torch.stack(coord_list)
    mean_tensor = torch.mean(coord_list, dim=0)
    std_tensor = torch.var(coord_list, dim = 0)
  
  G = nx.Graph()
  num_node = 0
  node_dict = {} # Hold all node coodinates
  for i in range(x_nodes):
      for j in range(y_nodes):
          coords = torch.tensor([float(i*side_x), float(j*side_y)]) - mean_tensor
          coords = coords/ std_tensor

          G.add_node(num_node, features = coords)
          node_dict[num_node] = coords
          t = torch.linspace(0, 1, subdivisions).reshape(-1, 1) #This will help us sample subnodes
          if j > 0:
              node_start = coords
              node_end = node_dict[num_node-1]
              subnodes = node_start + t*(node_end - node_start)
              G.add_edge(num_node, num_node-1)
              G.add_edge(num_node -1, num_node, subnodes = subnodes.flip(dims=(0,)))

          if i > 0:
              node_start = coords
              node_end = node_dict[num_node - y_nodes]
              subnodes = node_start + t*(node_end - node_start)

              G.add_edge(num_node, num_node - y_nodes)
              G.add_edge(num_node - y_nodes, num_node, subnodes = subnodes.flip(dims=(0,)))
          num_node += 1
  
  return G
    
def convert_nuplan_to_networkx(normalized = True):
  '''
  This function reads nuplan in the format that works for the model.
  normalized: Bool, we set this option to normalize the nodes, by default it will be true
  '''
  graphs = []
  maps = get_data_nuplan()

  for map in maps:
    G = nx.Graph()

    num_node = 0
    for node in map.nodes:
      if normalized:
         node = node / 32 # We know that nuplan data has nodes from -32 to 32
      G.add_node(num_node, features = torch.from_numpy(node).to(torch.float32))
      num_node += 1
    
    for edge_num in range(map.connections.shape[0]):
       start_node, end_node = map.connections[edge_num]
       subnodes = torch.from_numpy(map.subnodes[edge_num])
       if normalized:
          subnodes = subnodes / 32 # We know that nuplan data has nodes from -32 to 32

       G.add_edge(start_node, end_node, subnodes = subnodes)
       G.add_edge(end_node, start_node)
    
    graphs.append(G)
  
  return graphs
  

def create_graphs(graph_type, seed=1234):
  npr = np.random.RandomState(seed)
  ### load datasets
  graphs = []
  # synthetic graphs
  if graph_type == 'grid':
    graphs = []
    for i in range(10, 20):
      for j in range(10, 20):
        graphs.append(nx.grid_2d_graph(i, j))
   
  elif graph_type == 'grid_small':
    '''
      These were created mainly for testing base cases, the user can use the graph_type = 'grid'
      for general testing
    '''
    graphs = []
    for _ in range(50):
      graphs.append(nx.grid_2d_graph(5, 5))

  elif graph_type == 'grid_embed':
    '''
      Modified grid where each node has a coordinate attached to it.
    '''
    graphs = []
    for _ in range(35):
        graphs.append(create_grid_with_embed(5,5,normalized=True))
        # graphs.append(create_grid_with_embed(3,3,normalized=False))
  
  elif graph_type == 'subnode':
    '''
      Here Nodes have coordinates, as well we have subnodes with coordinates. 
      These subnodes helps the roads be curved.
      This is still used for debugging.
    '''
    graphs = []
    for _ in range(35):
        graphs.append(create_subnode_with_embed(5,5, subdivisions=20, normalized=True))
        # graphs.append(create_subnode_with_embed(4,4, subdivisions=20, normalized=True))
  
  elif graph_type == 'nuplan':
    output = convert_nuplan_to_networkx()
    graphs = []
    for i in range(len(output)):
      graphs.append(output[i])

  num_nodes = [gg.number_of_nodes() for gg in graphs]
  num_edges = [gg.number_of_edges() for gg in graphs]
  print('max # nodes = {} || mean # nodes = {}'.format(max(num_nodes), np.mean(num_nodes)))
  print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges)))
   
  return graphs

