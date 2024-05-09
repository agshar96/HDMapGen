import torch
import time
import os
import pickle
import glob
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from utils.data_helper import *


class GRANData(object):

  def __init__(self, config, graphs, tag='train'):
    self.config = config
    self.data_path = config.dataset.data_path
    self.model_name = config.model.name
    self.max_num_nodes = config.model.max_num_nodes
    self.block_size = config.model.block_size
    self.stride = config.model.sample_stride

    self.graphs = graphs
    self.num_graphs = len(graphs)
    self.npr = np.random.RandomState(config.seed)
    self.node_order = config.dataset.node_order
    self.num_canonical_order = config.model.num_canonical_order
    self.tag = tag
    self.num_fwd_pass = config.dataset.num_fwd_pass
    self.is_sample_subgraph = config.dataset.is_sample_subgraph
    self.num_subgraph_batch = config.dataset.num_subgraph_batch
    self.is_overwrite_precompute = config.dataset.is_overwrite_precompute

    if self.is_sample_subgraph:
      assert self.num_subgraph_batch > 0

    self.save_path = os.path.join(
        self.data_path, '{}_{}_{}_{}_{}_{}_{}_precompute'.format(
            config.model.name, config.dataset.name, tag, self.block_size,
            self.stride, self.num_canonical_order, self.node_order))
    if self.config.dataset.has_node_feat:
      self.embed_save_path = self.save_path + '_embed'
    
    if self.config.dataset.has_sub_nodes:
      self.subnode_save_path = self.save_path + '_subnode'

    if not os.path.isdir(self.save_path) or self.is_overwrite_precompute:
      self.file_names = []
      if not os.path.isdir(self.save_path):
        os.makedirs(self.save_path)
        if self.config.dataset.has_node_feat:
          os.makedirs(self.embed_save_path)
        if self.config.dataset.has_sub_nodes:
          os.makedirs(self.subnode_save_path)

      self.config.dataset.save_path = self.save_path
      for index in tqdm(range(self.num_graphs)):
        G = self.graphs[index]
        # If embeddings or subnodes are turned-off 'None' values are returned
        data, embeddings, subnodes = \
            self._get_graph_data(G, has_node_embed = config.dataset.has_node_feat,
                                has_subnode=config.dataset.has_sub_nodes,
                                has_end_node=config.dataset.has_stop_node)

        tmp_path = os.path.join(self.save_path, '{}_{}.p'.format(tag, index))
        pickle.dump(data, open(tmp_path, 'wb'))
        if config.dataset.has_node_feat:
          tmp_path_embedding = os.path.join(self.embed_save_path, '{}_{}.p'.format(tag, index))
          pickle.dump(embeddings, open(tmp_path_embedding, 'wb'))
        if config.dataset.has_sub_nodes:
          tmp_path_subnode = os.path.join(self.subnode_save_path, '{}_{}.p'.format(tag, index))
          pickle.dump(subnodes, open(tmp_path_subnode, 'wb'))
        self.file_names += [tmp_path]
    else:
      self.file_names = glob.glob(os.path.join(self.save_path, '*.p'))

  def get_new_edge_data(self, G, adj, nodelist):
    # get edge data from original graph
    edges = G.edges()
    edge_subnode_dict = {}
    for u,v in edges:
        edge_subnode_dict[(u,v)] = G[u][v]['subnodes']
        edge_subnode_dict[(v,u)] = G[v][u]['subnodes'].flip(dims=(0,))

    '''
    Logic explained:
    If in our current arrangement, node 5 shifts to first position so the nodelist will look like
    [5, ...] thus, while in our current graph node 5 is represented with position 0. Nodelist
    gives the actual node position in the original graph. Thus, we use this to get a mapping from
    new_node_positions to their original_node_positions
    '''
    mapping = {}
    for new_node_pos, og_node_pos in zip(G.nodes(), nodelist):
      # mapping[old_node] = new_node
      mapping[new_node_pos] = og_node_pos
    
    # Make adjacency lower traingular
    adj_l = np.tril(adj)
    rows, cols = np.nonzero(adj_l)

    # Now, using the above mapping, assign the correct subnodes to the correct edge
    edge_subnode_out = {}
    for x,y in zip(rows, cols):
      old_x = mapping[x]
      old_y = mapping[y]
      edge_subnode_out[(x,y)] = edge_subnode_dict[(old_x, old_y)]
    
    return edge_subnode_out

  def create_end_subnode(self, edge_subnote_out_1, max_node, node_embed):
    # This adds subnode coordinates between added end node and all previous node. (As,
    # end node is connected to all other nodes)
    t = torch.linspace(0, 1, self.config.dataset.num_sub_nodes).reshape(-1, 1)
    for i in range(max_node+1):
      start_node = node_embed[max_node,1]['features']
      end_node = node_embed[i,1]['features']
      subnode_tmp = start_node + t*(end_node - start_node)

      edge_subnote_out_1[(max_node, i)] = subnode_tmp

    return edge_subnote_out_1

  def add_end_node(self, adj, node_embed = None, subnode= None):
    ## Adding node to adjacency
    max_node = adj.shape[0]
    adj_new = np.pad(adj, ((0,1),(0,1)), mode='constant', constant_values=1)
    if node_embed is None:
      return adj_new, None, None
    ## Adding to node embedding
    end_node_dict = {}
    end_node_dict['features'] = torch.tensor([-1.0,-1.0]) ## We say end node will have -1,-1 coordinate
    end_node_arr = np.array([max_node, end_node_dict], dtype=node_embed.dtype).reshape(1,-1)
    node_embed_new = np.concatenate((node_embed, end_node_arr), axis=0)

    if subnode is None:
      return adj_new, node_embed_new, None
    ## Adding to subnode array
    subnode_new = self.create_end_subnode(subnode, max_node, node_embed_new)

    return adj_new, node_embed_new, subnode_new

  def  _get_graph_data(self, G, has_node_embed=False, has_subnode=False, has_end_node=False):

    # Default return values
    adj_list, node_embed_list, subnode_list = None, None, None

    # The first ordering is the default ordering of the dataset
    node_degree_list = [(n, d) for n, d in G.degree()]

    adj_0 = np.array(nx.to_numpy_array(G))
    if has_node_embed:
      node_embed_0 = np.array(list(G.nodes(data=True)))
    if has_subnode:
      edge_subnote_out_0 = self.get_new_edge_data(G, adj_0, list(G.nodes()))

    ### Degree descent ranking
    # N.B.: largest-degree node may not be unique
    adj_1, node_embed_1, edge_subnote_out_1 = [None, None, None]
    degree_sequence = sorted(
        node_degree_list, key=lambda tt: tt[1], reverse=True)
    nodelist_1= [dd[0] for dd in degree_sequence]
    adj_1 = np.array(
        nx.to_numpy_array(G, nodelist=nodelist_1))
    
    if has_node_embed:
      ## This code finds index where node is equal to node in nodelist_1
      ## For finding the index we use np.argmax to return the first match
      ## node_embed_0 has structure [node_number, {features = node_coordinates}]
      node_embed_1 = np.array([node_embed_0[np.argmax(node_embed_0[:,0] == node)] for node in nodelist_1])

    if has_subnode:
      edge_subnote_out_1 = self.get_new_edge_data(G, adj_1, nodelist_1)
  
    if has_end_node:
      adj_1, node_embed_1, edge_subnote_out_1 = self.add_end_node(adj_1, node_embed_1, edge_subnote_out_1)

    ### Degree ascent ranking
    adj_2, node_embed_2, edge_subnote_out_2 = [None, None, None]
    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
    nodelist_2 = [dd[0] for dd in degree_sequence]
    adj_2 = np.array(
        nx.to_numpy_array(G, nodelist=nodelist_2))
    if has_node_embed:
      node_embed_2 = np.array([node_embed_0[np.argmax(node_embed_0[:,0] == node)] for node in nodelist_2])

    if has_subnode:
      edge_subnote_out_2 = self.get_new_edge_data(G, adj_2, nodelist_2)
    
    if has_end_node:
      adj_2, node_embed_2, edge_subnote_out_2 = self.add_end_node(adj_2, node_embed_2, edge_subnote_out_2)

    ### BFS & DFS from largest-degree node
    adj_3, node_embed_3, edge_subnote_out_3 = [None, None, None]
    adj_4, node_embed_4, edge_subnote_out_4 = [None, None, None]
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]

    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    node_list_bfs = []
    node_list_dfs = []
    for ii in range(len(CGs)):
      node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
      degree_sequence = sorted(
          node_degree_list, key=lambda tt: tt[1], reverse=True)

      bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
      dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

      node_list_bfs += list(bfs_tree.nodes())
      node_list_dfs += list(dfs_tree.nodes())

    adj_3 = np.array(nx.to_numpy_array(G, nodelist=node_list_bfs))
    if has_node_embed:
      node_embed_3 = np.array([node_embed_0[np.argmax(node_embed_0[:,0] == node)] for node in node_list_bfs])
    if has_subnode:
      edge_subnote_out_3 = self.get_new_edge_data(G, adj_3, node_list_bfs)

    if has_end_node:
      adj_3, node_embed_3, edge_subnote_out_3 = self.add_end_node(adj_3, node_embed_3, edge_subnote_out_3)

    adj_4 = np.array(nx.to_numpy_array(G, nodelist=node_list_dfs))
    if has_node_embed:
      node_embed_4 = np.array([node_embed_0[np.argmax(node_embed_0[:,0] == node)] for node in node_list_dfs])
    if has_subnode:
      edge_subnote_out_4 = self.get_new_edge_data(G, adj_4, node_list_dfs)
    
    if has_end_node:
      adj_4, node_embed_4, edge_subnote_out_4 = self.add_end_node(adj_4, node_embed_4, edge_subnote_out_4)

    ### k-core
    adj_5, node_embed_5, edge_subnote_out_5 = [None, None, None]
    num_core = nx.core_number(G)
    core_order_list = sorted(list(set(num_core.values())), reverse=True)
    degree_dict = dict(G.degree())
    core_to_node = defaultdict(list)
    for nn, kk in num_core.items():
      core_to_node[kk] += [nn]

    node_list = []
    for kk in core_order_list:
      sort_node_tuple = sorted(
          [(nn, degree_dict[nn]) for nn in core_to_node[kk]],
          key=lambda tt: tt[1],
          reverse=True)
      node_list += [nn for nn, dd in sort_node_tuple]

    adj_5 = np.array(nx.to_numpy_array(G, nodelist=node_list))
    if has_node_embed:
      node_embed_5 = np.array([node_embed_0[np.argmax(node_embed_0[:,0] == node)] for node in node_list])

    if has_subnode:
      edge_subnote_out_5 = self.get_new_edge_data(G, adj_5, node_list)

    if has_end_node:
      adj_5, node_embed_5, edge_subnote_out_5 = self.add_end_node(adj_5, node_embed_5, edge_subnote_out_5)

    if self.num_canonical_order == 5:
      adj_list = [adj_0, adj_1, adj_3, adj_4, adj_5]
      if has_node_embed:
        node_embed_list = [node_embed_0, node_embed_1, node_embed_2,
                          node_embed_3, node_embed_4, node_embed_5]
      if has_subnode:
        subnode_list = [edge_subnote_out_0, edge_subnote_out_1, edge_subnote_out_2,
                        edge_subnote_out_3, edge_subnote_out_4, edge_subnote_out_5]
    else:
      if self.node_order == 'degree_decent':
        adj_list = [adj_1]
        if has_node_embed:
          node_embed_list = [node_embed_1]
        if has_subnode:
          subnode_list = [edge_subnote_out_1]
      elif self.node_order == 'degree_accent':
        adj_list = [adj_2]
        if has_node_embed:
          node_embed_list = [node_embed_2]
        if has_subnode:
          subnode_list = [edge_subnote_out_2]
      elif self.node_order == 'BFS':
        adj_list = [adj_3]
        if has_node_embed:
          node_embed_list = [node_embed_3]
        if has_subnode:
          subnode_list = [edge_subnote_out_3]
      elif self.node_order == 'DFS':
        adj_list = [adj_4]
        if has_node_embed:
          node_embed_list = [node_embed_4]
        if has_subnode:
          subnode_list = [edge_subnote_out_4]
      elif self.node_order == 'k_core':
        adj_list = [adj_5]
        if has_node_embed:
          node_embed_list = [node_embed_5]
        if has_subnode:
          subnode_list = [edge_subnote_out_5]
      elif self.node_order == 'DFS+BFS':
        adj_list = [adj_4, adj_3]
        if has_node_embed:
          node_embed_list = [node_embed_4, node_embed_3]
        if has_subnode:
          subnode_list = [edge_subnote_out_4,edge_subnote_out_3]
      elif self.node_order == 'DFS+BFS+k_core':
        adj_list = [adj_4, adj_3, adj_5]
        if has_node_embed:
          node_embed_list = [node_embed_4, node_embed_3, node_embed_5]
        if has_subnode:
          subnode_list = [edge_subnote_out_4, edge_subnote_out_3, edge_subnote_out_5]
      elif self.node_order == 'DFS+BFS+k_core+degree_decent':
        adj_list = [adj_4, adj_3, adj_5, adj_1]
        if has_node_embed:
          node_embed_list = [node_embed_4, node_embed_3, node_embed_5, node_embed_1]
        if has_subnode:
          subnode_list = [edge_subnote_out_4, edge_subnote_out_3, edge_subnote_out_5,
                          edge_subnote_out_1]
      elif self.node_order == 'all':
        adj_list = [adj_4, adj_3, adj_5, adj_1, adj_0]
        if has_node_embed:
          node_embed_list = [node_embed_4, node_embed_3, node_embed_5, node_embed_1, node_embed_0]
        if has_subnode:
          subnode_list = [edge_subnote_out_4, edge_subnote_out_3, edge_subnote_out_5, edge_subnote_out_1, edge_subnote_out_0]
      else:
        adj_list = [adj_0]
        if has_node_embed:
          node_embed_list = [node_embed_0]
        if has_subnode:
          subnode_list = [edge_subnote_out_0]

    # print('number of nodes = {}'.format(adj_0.shape[0]))
    return adj_list, node_embed_list, subnode_list

  def __getitem__(self, index):
    K = self.block_size
    N = self.max_num_nodes
    S = self.stride

    # load graph
    adj_list = pickle.load(open(self.file_names[index], 'rb'))
    if self.config.dataset.has_node_feat:
      embed_path = os.path.join(self.embed_save_path,self.file_names[index].split("/")[-1]) 
      node_embed_list = pickle.load(open(embed_path, 'rb'))
    if self.config.dataset.has_sub_nodes:
      subnode_path = os.path.join(self.subnode_save_path,self.file_names[index].split("/")[-1])
      subnode_list = pickle.load(open(subnode_path, 'rb'))

    num_nodes = adj_list[0].shape[0]
    num_subgraphs = int(np.floor((num_nodes - K) / S) + 1)

    if self.is_sample_subgraph:
      if self.num_subgraph_batch < num_subgraphs:
        num_subgraphs_pass = int(
            np.floor(self.num_subgraph_batch / self.num_fwd_pass))
      else:
        num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))

      end_idx = min(num_subgraphs, self.num_subgraph_batch)
    else:
      num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))
      end_idx = num_subgraphs

    ### random permute subgraph
    rand_perm_idx = self.npr.permutation(num_subgraphs).tolist()

    start_time = time.time()
    data_batch = []
    for ff in range(self.num_fwd_pass):
      ff_idx_start = num_subgraphs_pass * ff
      if ff == self.num_fwd_pass - 1:
        ff_idx_end = end_idx
      else:
        ff_idx_end = (ff + 1) * num_subgraphs_pass

      rand_idx = rand_perm_idx[ff_idx_start:ff_idx_end]

      edges = []
      node_idx_gnn = []
      node_idx_feat = []
      if self.config.dataset.has_node_feat:
        node_embed_idx_gnn = []
        label_embed = []
      if self.config.dataset.has_sub_nodes:
        subnode_coords = []
        subnode_labels = []
      label = []
      subgraph_size = []
      subgraph_idx = []
      att_idx = []
      subgraph_count = 0

      for ii in range(len(adj_list)):
        # loop over different orderings
        adj_full = adj_list[ii]
        if self.config.dataset.has_sub_nodes:
          subnode_dict = subnode_list[ii]
        if self.config.dataset.has_node_feat:
          node_embed_output = node_embed_list[ii][:, 1]
          node_embed = np.array([item['features'].numpy() for item in node_embed_output])
        # adj_tril = np.tril(adj_full, k=-1)

        idx = -1
        for jj in range(0, num_nodes, S):
          # loop over different subgraphs
          idx += 1

          ### for each size-(jj+K) subgraph, we generate edges for the new block of K nodes
          if jj + K > num_nodes:
            break

          if idx not in rand_idx:
            continue

          ### get graph for GNN propagation
          adj_block = np.pad(
              adj_full[:jj, :jj], ((0, K), (0, K)),
              'constant',
              constant_values=1.0)  # assuming fully connected for the new block
          adj_block = np.tril(adj_block, k=-1)
          adj_block = adj_block + adj_block.transpose()
          adj_block = torch.from_numpy(adj_block).to_sparse()
          edges += [adj_block.coalesce().indices().long()]

          ### get attention index
          # exist node: 0
          # newly added node: 1, ..., K
          if jj == 0:
            att_idx += [np.arange(1, K + 1).astype(np.uint8)]
          else:
            att_idx += [
                np.concatenate([
                    np.zeros(jj).astype(np.uint8),
                    np.arange(1, K + 1).astype(np.uint8)
                ])
            ]

          ### get node feature index for GNN input
          # use inf to indicate the newly added nodes where input feature is zero
          if jj == 0:
            node_idx_feat += [np.ones(K) * np.inf]
          else:
            node_idx_feat += [
                np.concatenate([np.arange(jj) + ii * N,
                                np.ones(K) * np.inf])
            ]

          ### get node index for GNN output
          idx_row_gnn, idx_col_gnn = np.meshgrid(
              np.arange(jj, jj + K), np.arange(jj + K))
          idx_row_gnn = idx_row_gnn.reshape(-1, 1)
          idx_col_gnn = idx_col_gnn.reshape(-1, 1)
          node_idx_gnn += [
              np.concatenate([idx_row_gnn, idx_col_gnn],
                             axis=1).astype(np.int64)
          ]
          '''
          We make two arrays here, E = No.of edges
          For each edge produced we save the cooresponding start and end coordinate in subnode_coords
          subnode_coords shape Ex4 (2 for start, 2 for end). subnode_coords are input for subnode prediction.
          For each edge we also produce all the subnodes. Here, we assume 10 subnodes.
          subnode_labels: E x (2*num_sub_nodes) (since each subnode is 2D and 'num_sub_nodes' subnodes in total)
          subnode_labels are output by our neural network given first and last coordinate and graph features
          '''
          if self.config.dataset.has_sub_nodes:
            node_embed_start = node_embed[idx_row_gnn].reshape(idx_row_gnn.shape[0], -1)
            node_embed_end = node_embed[idx_col_gnn].reshape(idx_row_gnn.shape[0], -1)
            subnode_coords += [
              np.concatenate([node_embed_start, node_embed_end],
                             axis=1).astype(np.float32)
            ]
            subnode_tmp = []
            for x,y in zip(idx_row_gnn, idx_col_gnn):
              x = x.item()
              y = y.item()
              if (x,y) in subnode_dict:
                subnode_tmp += [subnode_dict[(x,y)].numpy()[None, :].reshape(-1, 2*self.config.dataset.num_sub_nodes).astype(np.float32)]
              elif (y,x) in subnode_dict:
                subnode_tmp += [subnode_dict[(y,x)].flip(dims=(0,)).numpy()[None,:].reshape(-1, 2*self.config.dataset.num_sub_nodes).astype(np.float32)]
              else:
                subnode_tmp += [np.zeros((1,2*self.config.dataset.num_sub_nodes)).astype(np.float32)]
            subnode_tmp = np.concatenate(subnode_tmp)
            subnode_labels += [subnode_tmp]
          ### get predict label
          label += [
              adj_full[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
          ]

          ### get node embedding labels for training
          if self.config.dataset.has_node_feat:
            idx_embed_gnn = np.arange(jj, jj+K)
            node_embed_idx_gnn += [idx_embed_gnn.astype(np.int64)] # This records idx of new nodes introduced
            label_embed += [
              node_embed[idx_embed_gnn].astype(np.float32)
            ]

          subgraph_size += [jj + K]
          subgraph_idx += [
              np.ones_like(label[-1]).astype(np.int64) * subgraph_count
          ]
          subgraph_count += 1

      ### adjust index basis for the selected subgraphs
      cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
      for ii in range(len(edges)):
        edges[ii] = edges[ii] + cum_size[ii]
        node_idx_gnn[ii] = node_idx_gnn[ii] + cum_size[ii]
        if self.config.dataset.has_node_feat:
          node_embed_idx_gnn[ii] = node_embed_idx_gnn[ii] + cum_size[ii]

      ### pack tensors
      data = {}
      data['adj'] = np.tril(np.stack(adj_list, axis=0), k=-1)
      #TO DO: This will currently work if you only use one means of traversal
      ## for instance just the DFS or the BFS but never multiple simulataneously
      if self.config.dataset.has_node_feat:
        data['node_embed'] = np.expand_dims(node_embed, axis=0)
        data['node_embed_idx_gnn'] = np.concatenate(node_embed_idx_gnn)
        data['label_embed'] = np.concatenate(label_embed)
      if self.config.dataset.has_sub_nodes:
        data['subnode_coords'] = np.concatenate(subnode_coords)
        data['subnode_labels'] = np.concatenate(subnode_labels)
      data['edges'] = torch.cat(edges, dim=1).t().long()
      data['node_idx_gnn'] = np.concatenate(node_idx_gnn)
      data['node_idx_feat'] = np.concatenate(node_idx_feat)
      data['label'] = np.concatenate(label)
      data['att_idx'] = np.concatenate(att_idx)
      data['subgraph_idx'] = np.concatenate(subgraph_idx)
      data['subgraph_count'] = subgraph_count
      data['num_nodes'] = num_nodes
      data['subgraph_size'] = subgraph_size
      data['num_count'] = sum(subgraph_size)
      data_batch += [data]

    end_time = time.time()

    return data_batch

  def __len__(self):
    return self.num_graphs

  def collate_fn(self, batch):
    assert isinstance(batch, list)
    start_time = time.time()
    batch_size = len(batch)
    N = self.max_num_nodes
    C = self.num_canonical_order
    batch_data = []

    for ff in range(self.num_fwd_pass):
      data = {}
      batch_pass = []
      for bb in batch:
        batch_pass += [bb[ff]]

      pad_size = [self.max_num_nodes - bb['num_nodes'] for bb in batch_pass]
      subgraph_idx_base = np.array([0] +
                                   [bb['subgraph_count'] for bb in batch_pass])
      subgraph_idx_base = np.cumsum(subgraph_idx_base)

      data['subgraph_idx_base'] = torch.from_numpy(
        subgraph_idx_base)

      data['num_nodes_gt'] = torch.from_numpy(
          np.array([bb['num_nodes'] for bb in batch_pass])).long().view(-1)

      data['adj'] = torch.from_numpy(
          np.stack(
              [
                  np.pad(
                      bb['adj'], ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
                      'constant',
                      constant_values=0.0) for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).float()  # B X C X N X N
      
      # if self.config.dataset.has_stop_node:
      #   end_node_adj = torch.ones((self.max_num_nodes), dtype=torch.float32)
      #   end_node_adj[-1:] = 0.0
      #   adj_tmp = torch.from_numpy(
      #     np.stack(
      #         [
      #             np.pad(
      #                 bb['adj'], ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
      #                 'constant',
      #                 constant_values=0.0) for ii, bb in enumerate(batch_pass)
      #         ],
      #         axis=0)).float()

      #   for ii,bb in enumerate(batch_pass):
      #     test = adj_tmp[ii, 0, bb['num_nodes'] -1, :]
      #     adj_tmp[ii, 0, bb['num_nodes'] -1, :] = end_node_adj
        
      #   data['adj'] = adj_tmp
      
      if self.config.dataset.has_node_feat:
        data['node_embed'] = torch.from_numpy(
            np.stack(
                [
                    np.pad(
                        bb['node_embed'], ((0, 0), (0, pad_size[ii]), (0, 0)),
                        'constant',
                        constant_values=-1.0) for ii, bb in enumerate(batch_pass)
                ],
                axis=0)).float() # B X N X 2

      idx_base = np.array([0] + [bb['num_count'] for bb in batch_pass])
      idx_base = np.cumsum(idx_base)

      data['edges'] = torch.cat(
          [bb['edges'] + idx_base[ii] for ii, bb in enumerate(batch_pass)],
          dim=0).long()

      data['node_idx_gnn'] = torch.from_numpy(
          np.concatenate(
              [
                  bb['node_idx_gnn'] + idx_base[ii]
                  for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).long()
      
      if self.config.dataset.has_sub_nodes:
        data['subnode_coords'] = torch.from_numpy(
          np.concatenate(
            [
                bb['subnode_coords'] for bb in batch_pass
            ],
            axis=0)).float()
          
        
        data['subnode_labels'] = torch.from_numpy(
          np.concatenate(
            [
              bb['subnode_labels'] for bb in batch_pass
            ],
          axis=0)).float()
        

      if self.config.dataset.has_node_feat:
        data['node_embed_idx_gnn'] = torch.from_numpy(
          np.concatenate(
              [
                  bb['node_embed_idx_gnn'] + idx_base[ii]
                  for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).long()

      data['att_idx'] = torch.from_numpy(
          np.concatenate([bb['att_idx'] for bb in batch_pass], axis=0)).long()

      # shift one position for padding 0-th row feature in the model
      node_idx_feat = np.concatenate(
          [
              bb['node_idx_feat'] + ii * C * N
              for ii, bb in enumerate(batch_pass)
          ],
          axis=0) + 1
      node_idx_feat[np.isinf(node_idx_feat)] = 0
      node_idx_feat = node_idx_feat.astype(np.int64)
      data['node_idx_feat'] = torch.from_numpy(node_idx_feat).long()

      data['label'] = torch.from_numpy(
          np.concatenate([bb['label'] for bb in batch_pass])).float()

      if self.config.dataset.has_node_feat:
        data['label_embed'] = torch.from_numpy(
          np.concatenate([bb['label_embed'] for bb in batch_pass])).float()

      data['subgraph_idx'] = torch.from_numpy(
          np.concatenate([
              bb['subgraph_idx'] + subgraph_idx_base[ii]
              for ii, bb in enumerate(batch_pass)
          ])).long()

      batch_data += [data]

    end_time = time.time()
    # print('collate time = {}'.format(end_time - start_time))

    return batch_data
