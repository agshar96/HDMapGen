import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import groupby
EPS = np.finfo(np.float32).eps

__all__ = ['GRANMixtureBernoulli']


class GNN(nn.Module):

  def __init__(self,
               msg_dim,
               node_state_dim,
               edge_feat_dim,
               num_prop=1,
               num_layer=1,
               has_attention=True,
               att_hidden_dim=128,
               has_residual=False,
               has_graph_output=False,
               output_hidden_dim=128,
               graph_output_dim=None):
    super(GNN, self).__init__()
    self.msg_dim = msg_dim
    self.node_state_dim = node_state_dim
    self.edge_feat_dim = edge_feat_dim
    self.num_prop = num_prop
    self.num_layer = num_layer
    self.has_attention = has_attention
    self.has_residual = has_residual
    self.att_hidden_dim = att_hidden_dim
    self.has_graph_output = has_graph_output
    self.output_hidden_dim = output_hidden_dim
    self.graph_output_dim = graph_output_dim

    self.update_func = nn.ModuleList([
        nn.GRUCell(input_size=self.msg_dim, hidden_size=self.node_state_dim)
        for _ in range(self.num_layer)
    ])

    self.msg_func = nn.ModuleList([
        nn.Sequential(
            *[
                nn.Linear(self.node_state_dim + self.edge_feat_dim,
                          self.msg_dim),
                nn.ReLU(),
                nn.Linear(self.msg_dim, self.msg_dim)
            ]) for _ in range(self.num_layer)
    ])

    if self.has_attention:
      self.att_head = nn.ModuleList([
          nn.Sequential(
              *[
                  nn.Linear(self.node_state_dim + self.edge_feat_dim,
                            self.att_hidden_dim),
                  nn.ReLU(),
                  nn.Linear(self.att_hidden_dim, self.msg_dim),
                  nn.Sigmoid()
              ]) for _ in range(self.num_layer)
      ])

    if self.has_graph_output:
      self.graph_output_head_att = nn.Sequential(*[
          nn.Linear(self.node_state_dim, self.output_hidden_dim),
          nn.ReLU(),
          nn.Linear(self.output_hidden_dim, 1),
          nn.Sigmoid()
      ])

      self.graph_output_head = nn.Sequential(
          *[nn.Linear(self.node_state_dim, self.graph_output_dim)])

  def _prop(self, state, edge, edge_feat, layer_idx=0):
    ### compute message
    state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]
    if self.edge_feat_dim > 0:
      edge_input = torch.cat([state_diff, edge_feat], dim=1)
    else:
      edge_input = state_diff

    msg = self.msg_func[layer_idx](edge_input)

    ### attention on messages
    if self.has_attention:
      att_weight = self.att_head[layer_idx](edge_input)
      msg = msg * att_weight

    ### aggregate message by sum
    state_msg = torch.zeros(state.shape[0], msg.shape[1]).to(state.device)
    scatter_idx = edge[:, [1]].expand(-1, msg.shape[1])
    state_msg = state_msg.scatter_add(0, scatter_idx, msg)

    ### state update
    state = self.update_func[layer_idx](state_msg, state)
    return state

  def forward(self, node_feat, edge, edge_feat, graph_idx=None):
    """
      N.B.: merge a batch of graphs as a single graph

      node_feat: N X D, node feature
      edge: M X 2, edge indices
      edge_feat: M X D', edge feature
      graph_idx: N X 1, graph indices
    """

    state = node_feat
    prev_state = state
    for ii in range(self.num_layer):
      if ii > 0:
        state = F.relu(state)

      for jj in range(self.num_prop):
        state = self._prop(state, edge, edge_feat=edge_feat, layer_idx=ii)

    if self.has_residual:
      state = state + prev_state

    if self.has_graph_output:
      num_graph = graph_idx.max() + 1
      node_att_weight = self.graph_output_head_att(state)
      node_output = self.graph_output_head(state)

      # weighted average
      reduce_output = torch.zeros(num_graph,
                                  node_output.shape[1]).to(node_feat.device)
      reduce_output = reduce_output.scatter_add(0,
                                                graph_idx.unsqueeze(1).expand(
                                                    -1, node_output.shape[1]),
                                                node_output * node_att_weight)

      const = torch.zeros(num_graph).to(node_feat.device)
      const = const.scatter_add(
          0, graph_idx, torch.ones(node_output.shape[0]).to(node_feat.device))

      reduce_output = reduce_output / const.view(-1, 1)

      return reduce_output
    else:
      return state


class GRANMixtureBernoulli(nn.Module):
  """ Graph Recurrent Attention Networks """

  def __init__(self, config):
    super(GRANMixtureBernoulli, self).__init__()
    self.config = config
    self.device = config.device
    self.max_num_nodes = config.model.max_num_nodes
    self.hidden_dim = config.model.hidden_dim
    self.is_sym = config.model.is_sym
    self.block_size = config.model.block_size
    self.sample_stride = config.model.sample_stride
    self.num_GNN_prop = config.model.num_GNN_prop
    self.num_GNN_layers = config.model.num_GNN_layers
    self.edge_weight = config.model.edge_weight if hasattr(
        config.model, 'edge_weight') else 1.0
    self.dimension_reduce = config.model.dimension_reduce
    self.has_attention = config.model.has_attention
    self.num_canonical_order = config.model.num_canonical_order
    self.output_dim = 1
    self.num_mix_component = config.model.num_mix_component
    self.has_rand_feat = False # use random feature instead of 1-of-K encoding
    self.att_edge_dim = 16 #64
    if config.dataset.has_node_feat:
      self.node_embedding_in_dim = config.model.node_embedding_dim
    if config.dataset.has_sub_nodes:
      self.subnode_out_dim = 2 * config.dataset.num_sub_nodes
      # self.node_embedding_out_dim = 16 ## Currently not added to config

    self.output_theta = nn.Sequential(
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.output_dim * self.num_mix_component))

    self.output_alpha = nn.Sequential(
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.num_mix_component))
    
    if config.dataset.has_node_feat:
      self.output_gmm_weights = nn.Sequential(
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.num_mix_component)
      )

      self.output_gmm_mv = nn.Sequential( # mv stands for mean and variance
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        ## Here the first node_embed_in_dim* num_mixture indices are for mean and the last node_embed_in_dim * num_mixture are for variance.
        ## Following this convention from mixture density networks. We also assume covariance to be diagonal.
        nn.Linear(self.hidden_dim, 2 * self.node_embedding_in_dim * self.num_mix_component) ## 2 * node_in because we have mean and variance so 2
      )  
    
    if config.dataset.has_sub_nodes:
      self.output_subnode_decoder = nn.Sequential(
        nn.Linear(4 + 2*self.hidden_dim, self.hidden_dim), # 4 because 2 2D coordinates, 2*hidden as we take hidden embedding of start and end
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.subnode_out_dim)
      )

    if self.dimension_reduce:
      self.embedding_dim = config.model.embedding_dim
      # Change to make embeddings concatenated
      if config.dataset.has_node_feat:
        self.embedding_dim = int(self.embedding_dim / 2)
        self.node_embedding_out_dim = self.embedding_dim

      self.decoder_input = nn.Sequential(
          nn.Linear(self.max_num_nodes, self.embedding_dim))
      if config.dataset.has_node_feat:
        self.decoder_embedding = nn.Sequential(
          nn.Linear(self.node_embedding_in_dim, self.node_embedding_out_dim)
        )
        # self.decoder_combine = nn.Sequential(
        #   nn.Linear(self.node_embedding_out_dim + self.embedding_dim, self.hidden_dim)
        # )
    else:
      self.embedding_dim = self.max_num_nodes

    self.decoder = GNN(
        msg_dim=self.hidden_dim,
        node_state_dim=self.hidden_dim,
        edge_feat_dim=2 * self.att_edge_dim,
        num_prop=self.num_GNN_prop,
        num_layer=self.num_GNN_layers,
        has_attention=self.has_attention)

    ### Loss functions
    pos_weight = torch.ones([1]) * self.edge_weight
    self.adj_loss_func = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction='none')

  def _inference(self,
                 A_pad=None,
                 edges=None,
                 node_idx_gnn=None,
                 node_idx_feat=None,
                 att_idx=None,
                 node_embed_pad = None,
                 node_embed_idx_gnn = None,
                 subnode_coords = None):
    """ generate adj in row-wise auto-regressive fashion """

    B, C, N_max, _ = A_pad.shape
    H = self.hidden_dim
    K = self.block_size
    A_pad = A_pad.view(B * C * N_max, -1)
    if self.config.dataset.has_node_feat:
      node_embed_pad = node_embed_pad.view(B*C*N_max, -1)

    # Setting default return values
    log_theta, log_alpha, embed_log_pis, embed_mv, output_subnodes = 0,0,0,0,0
    
    # Add noise for data augmentation purposes. It's done here cause it lesser changes than dataloader
    if self.config.dataset.is_noisy:
      noisy_std = self.config.dataset.noise_std
      if self.config.dataset.has_node_feat:
        node_embed_pad += (torch.randn(node_embed_pad.size()) * noisy_std).to(self.device)
      if self.config.dataset.has_sub_nodes:
        subnode_coords += (torch.randn(subnode_coords.size()) * noisy_std).to(self.device)


    if self.dimension_reduce:
      if self.config.dataset.has_node_feat:
        adj_feat = self.decoder_input(A_pad)  # BCN_max X H
        embed_feat = self.decoder_embedding(node_embed_pad)
        node_feat = torch.cat((adj_feat, embed_feat), dim=1)
      else:
        node_feat = self.decoder_input(A_pad)

    else:
      node_feat = A_pad  # BCN_max X N_max

    ### GNN inference
    # pad zero as node feature for newly generated nodes (1st row)
    node_feat = F.pad(
        node_feat, (0, 0, 1, 0), 'constant', value=0.0)  # (BCN_max + 1) X N_max

    # create symmetry-breaking edge feature for the newly generated nodes
    att_idx = att_idx.view(-1, 1)

    if self.has_rand_feat:
      # create random feature
      att_edge_feat = torch.zeros(edges.shape[0],
                                  2 * self.att_edge_dim).to(node_feat.device)
      idx_new_node = (att_idx[[edges[:, 0]]] >
                      0).long() + (att_idx[[edges[:, 1]]] > 0).long()
      idx_new_node = idx_new_node.byte().squeeze()
      att_edge_feat[idx_new_node, :] = torch.randn(
          idx_new_node.long().sum(),
          att_edge_feat.shape[1]).to(node_feat.device)
    else:
      # create one-hot feature
      att_edge_feat = torch.zeros(edges.shape[0],
                                  2 * self.att_edge_dim).to(node_feat.device)
      # scatter with empty index seems to cause problem on CPU but not on GPU
      att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
      att_edge_feat = att_edge_feat.scatter(
          1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

    # GNN inference
    # N.B.: node_feat is shared by multiple subgraphs within the same batch
    node_state = self.decoder(
        node_feat[node_idx_feat], edges, edge_feat=att_edge_feat)

    ### Pairwise predict edges
    diff = node_state[node_idx_gnn[:, 0], :] - node_state[node_idx_gnn[:, 1], :]

    log_theta = self.output_theta(diff)  # B X (tt+K)K
    log_alpha = self.output_alpha(diff)  # B X (tt+K)K
    if self.config.dataset.has_node_feat:
      node_embed_out = node_state[node_embed_idx_gnn]
      embed_log_pis = self.output_gmm_weights(node_embed_out)
      embed_mv = self.output_gmm_mv(node_embed_out) ## mv stands for mean and variance
      
    if self.config.dataset.has_sub_nodes:
      decoder_input = torch.cat( (subnode_coords, node_state[node_idx_gnn[:, 0], :], 
                                  node_state[node_idx_gnn[:, 1],:]), dim=1)
      output_subnodes = self.output_subnode_decoder(decoder_input)

    log_theta = log_theta.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K
    log_alpha = log_alpha.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K

    return log_theta, log_alpha, embed_log_pis, embed_mv, output_subnodes


  def _sampling(self, B):
    """ generate adj in row-wise auto-regressive fashion """
    ## Default return values
    A, node_embed, subnode_coords, theta_ret = None, None, None, None
    with torch.no_grad():

      K = self.block_size
      S = self.sample_stride
      H = self.hidden_dim
      N = self.max_num_nodes
      mod_val = (N - K) % S
      if mod_val > 0:
        N_pad = N - K - mod_val + int(np.ceil((K + mod_val) / S)) * S
      else:
        N_pad = N

      A = torch.zeros(B, N_pad, N_pad).to(self.device)
      if self.config.test.animated_vis:
        ## We need to return theta in this case
        theta_ret = torch.zeros(B, N_pad, N_pad).to(self.device)
      if self.config.dataset.has_node_feat:
        node_embed = torch.zeros(B, N_pad, self.node_embedding_in_dim).to(self.device)
      if self.config.dataset.has_sub_nodes:
        subnode_coords = torch.zeros(B, N_pad*N_pad, 2*self.config.dataset.num_sub_nodes)
      dim_input = self.embedding_dim if self.dimension_reduce else self.max_num_nodes

      ### cache node state for speed up
      if self.config.dataset.has_node_feat:
        node_state = torch.zeros(B, N_pad, 2*dim_input).to(self.device)
      else:
        node_state = torch.zeros(B, N_pad, dim_input).to(self.device)
      # if self.config.dataset.has_node_feat:
      #   embed_state = torch.zeros(B, N_pad, self.node_embedding_out_dim).to(self.device)
      for ii in range(0, N_pad, S):
        # for ii in range(0, 3530, S):
        jj = ii + K
        if jj > N_pad:
          break

        # reset to discard overlap generation
        A[:, ii:, :] = .0
        A = torch.tril(A, diagonal=-1)

        if ii >= K:
          if self.dimension_reduce:
            if self.config.dataset.has_node_feat:
              adj_state = self.decoder_input(A[:, ii - K:ii, :N])
              embed_state = self.decoder_embedding(node_embed[:, ii - K:ii, :])
              node_state[:, ii - K:ii, :] = torch.cat((adj_state, embed_state), dim=-1)
            else:
              node_state[:, ii - K:ii, :] = self.decoder_input(A[:, ii - K:ii, :N])
          else:
            node_state[:, ii - K:ii, :] = A[:, ii - S:ii, :N]
        else:
          if self.dimension_reduce:
            if self.config.dataset.has_node_feat:
              adj_state = self.decoder_input(A[:, :ii, :N])
              embed_state = self.decoder_embedding(node_embed[:, :ii, :])
              node_state[:, :ii, :] = torch.cat((adj_state, embed_state), dim=-1)
            else:
              node_state[:, :ii, :] = self.decoder_input(A[:, :ii, :N])
          else:
            node_state[:, :ii, :] = A[:, ii - S:ii, :N]

        node_state_in = F.pad(
            node_state[:, :ii, :], (0, 0, 0, K), 'constant', value=.0)

        ### GNN propagation
        adj = F.pad(
            A[:, :ii, :ii], (0, K, 0, K), 'constant', value=1.0)  # B X jj X jj
        adj = torch.tril(adj, diagonal=-1)
        adj = adj + adj.transpose(1, 2)
        edges = [
            adj[bb].to_sparse().coalesce().indices() + bb * adj.shape[1]
            for bb in range(B)
        ]
        edges = torch.cat(edges, dim=1).t()

        att_idx = torch.cat([torch.zeros(ii).long(),
                              torch.arange(1, K + 1)]).to(self.device)
        att_idx = att_idx.view(1, -1).expand(B, -1).contiguous().view(-1, 1)

        if self.has_rand_feat:
          # create random feature
          att_edge_feat = torch.zeros(edges.shape[0],
                                      2 * self.att_edge_dim).to(self.device)
          idx_new_node = (att_idx[[edges[:, 0]]] >
                          0).long() + (att_idx[[edges[:, 1]]] > 0).long()
          idx_new_node = idx_new_node.byte().squeeze()
          att_edge_feat[idx_new_node, :] = torch.randn(
              idx_new_node.long().sum(), att_edge_feat.shape[1]).to(self.device)
        else:
          # create one-hot feature
          att_edge_feat = torch.zeros(edges.shape[0],
                                      2 * self.att_edge_dim).to(self.device)
          att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
          att_edge_feat = att_edge_feat.scatter(
              1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

        node_state_out = self.decoder(
            node_state_in.view(-1, H), edges, edge_feat=att_edge_feat)
        node_state_out = node_state_out.view(B, jj, -1)

        idx_row, idx_col = np.meshgrid(np.arange(ii, jj), np.arange(jj))
        idx_row = torch.from_numpy(idx_row.reshape(-1)).long().to(self.device)
        idx_col = torch.from_numpy(idx_col.reshape(-1)).long().to(self.device)
        if self.config.dataset.has_node_feat:
          idx_node = torch.from_numpy(np.arange(ii, jj)).long().to(self.device)

        diff = node_state_out[:,idx_row, :] - node_state_out[:,idx_col, :]  # B X (ii+K)K X H
        diff = diff.view(-1, node_state.shape[2])
        log_theta = self.output_theta(diff)
        log_alpha = self.output_alpha(diff)

        if self.config.dataset.has_node_feat:
          # node_embed_out = self.output_embed(
          #   node_state_out[:,idx_node,:].view(-1, node_state.shape[2])
          # ).view(B, -1, self.node_embedding_in_dim)
          output_log_pis = self.output_gmm_weights(
            node_state_out[:,idx_node,:].view(-1, node_state.shape[2]))
          output_mv = self.output_gmm_mv(
            node_state_out[:,idx_node,:].view(-1, node_state.shape[2])
          )
          log_pi, mu, sigma = self.rearrange_embed_out(output_log_pis, output_mv)
          cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
          rvs = torch.rand(len(output_log_pis), 1).to(self.device)
          rand_pi = torch.searchsorted(cum_pi, rvs)
          rand_normal = torch.randn_like(mu) * sigma + mu
          node_embed_out = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
          node_embed_out = node_embed_out.reshape(B, -1, self.node_embedding_in_dim)
          node_embed[:, ii:jj, :] = node_embed_out
        
        if self.config.dataset.has_sub_nodes:
          start_node = node_embed[:, idx_row, :]
          end_node = node_embed[:, idx_col, :]

          subnode_in = torch.cat((start_node, end_node, node_state_out[:,idx_row, :], node_state_out[:,idx_col, :])
                                  , dim = -1)
          subnode_in = subnode_in.reshape(-1, 2*self.node_embedding_in_dim + 2*H)
          subnode_coord_out = self.output_subnode_decoder(subnode_in).reshape(B, -1, 2*self.config.dataset.num_sub_nodes)

        log_theta = log_theta.view(B, -1, K, self.num_mix_component)  # B X K X (ii+K) X L
        log_theta = log_theta.transpose(1, 2)  # B X (ii+K) X K X L

        log_alpha = log_alpha.view(B, -1, self.num_mix_component)  # B X K X (ii+K)
        prob_alpha = F.softmax(log_alpha.mean(dim=1), -1)
        alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()

        prob = []
        for bb in range(B):
          prob += [torch.sigmoid(log_theta[bb, :, :, alpha[bb]])]

        prob = torch.stack(prob, dim=0)
        if self.config.dataset.has_sub_nodes:
          # Doing this because input was also scaled so it makes sense to do this
          prob_test = prob.reshape(-1,1)
          subnode_coord_out_test = subnode_coord_out.reshape(-1, 2*self.config.dataset.num_sub_nodes)
          # subnode_coord_out = prob.reshape(-1) * subnode_coord_out.reshape(-1, 2*self.config.dataset.num_sub_nodes)
          subnode_coord_out_test = prob_test * subnode_coord_out_test
          subnode_coords[:, ii*N_pad:ii*N_pad + subnode_coord_out.shape[1], :] = subnode_coord_out_test.reshape(B, -1, 2*self.config.dataset.num_sub_nodes)

        if self.config.test.animated_vis:
          theta_ret[:, ii:jj, :jj] = prob[:, :jj - ii, :]

        A[:, ii:jj, :jj] = torch.bernoulli(prob[:, :jj - ii, :])


      ### make it symmetric
      if self.is_sym:
        A = torch.tril(A, diagonal=-1)
        A = A + A.transpose(1, 2)
      
      return A, node_embed, subnode_coords, theta_ret

  '''
  These two functions have been copied from 'mixture density network' code
  and, they perform GMM loss based on network output
  '''
  def rearrange_embed_out(self, embed_log_pis, embed_mv, eps=1e-6):
    log_pi = torch.log_softmax(embed_log_pis, dim=-1)
    mu = embed_mv[..., :self.node_embedding_in_dim * self.num_mix_component]
    sigma = embed_mv[..., self.node_embedding_in_dim * self.num_mix_component:]
    sigma = torch.exp(sigma + eps)

    mu = mu.reshape(-1, self.num_mix_component, self.node_embedding_in_dim)
    sigma = sigma.reshape(-1, self.num_mix_component, self.node_embedding_in_dim)

    return log_pi, mu, sigma

  def loss_embed(self, embed_log_pis, embed_mv, label_embed):
        log_pi, mu, sigma = self.rearrange_embed_out(embed_log_pis, embed_mv)
        z_score = (label_embed.unsqueeze(1) - mu) / sigma
        normal_loglik = (
            -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            -torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik

  '''
  Following are helper functions that make central functions less cluttered
  '''
  def dict2var(self, input_dict):
    is_sampling = input_dict[
        'is_sampling'] if 'is_sampling' in input_dict else False
    batch_size = input_dict[
        'batch_size'] if 'batch_size' in input_dict else None
    A_pad = input_dict['adj'] if 'adj' in input_dict else None

    node_embed_pad = input_dict['node_embed'] if 'node_embed' in input_dict else None
    node_embed_idx_gnn = input_dict['node_embed_idx_gnn'] if 'node_embed_idx_gnn' in input_dict else None
    label_embed = input_dict['label_embed'] if 'label_embed' in input_dict else None

    subnode_labels = input_dict['subnode_labels'] if 'subnode_labels' in input_dict else None
    subnode_coords = input_dict['subnode_coords'] if 'subnode_coords' in input_dict else None

    node_idx_gnn = input_dict[
        'node_idx_gnn'] if 'node_idx_gnn' in input_dict else None
    node_idx_feat = input_dict[
        'node_idx_feat'] if 'node_idx_feat' in input_dict else None
    att_idx = input_dict['att_idx'] if 'att_idx' in input_dict else None
    subgraph_idx = input_dict[
        'subgraph_idx'] if 'subgraph_idx' in input_dict else None
    edges = input_dict['edges'] if 'edges' in input_dict else None
    label = input_dict['label'] if 'label' in input_dict else None
    num_nodes_pmf = input_dict[
        'num_nodes_pmf'] if 'num_nodes_pmf' in input_dict else None
    subgraph_idx_base = input_dict[
        "subgraph_idx_base"] if "subgraph_idx_base" in input_dict else None
    
    return is_sampling, batch_size, A_pad, node_embed_pad, node_embed_idx_gnn, label_embed \
          , subnode_coords, subnode_labels, node_idx_gnn, node_idx_feat, att_idx, subgraph_idx \
          , edges, label, num_nodes_pmf, subgraph_idx_base

  def get_losses(self, log_theta, log_alpha, subgraph_idx, subgraph_idx_base,
                 label, embed_log_pis, embed_mv, label_embed,
                 output_subnodes, subnode_labels):

    ## Initializing Losses. If a certain feature is not enabled, the loss = 0, for that
    adj_loss, subnode_loss, embed_loss = 0,0,0

    adj_loss = mixture_bernoulli_loss(label, log_theta, log_alpha,
                                        self.adj_loss_func, subgraph_idx, subgraph_idx_base,
                                        self.num_canonical_order)

    if self.config.dataset.has_node_feat:
      embed_loss = self.loss_embed(embed_log_pis, embed_mv, label_embed)
      embed_loss = embed_loss.mean()
      # print("Output embed shape: ", output_embed.shape, " label_embed shape: ", label_embed.shape)
      # print("embed loss: ", embed_loss)
    
    if self.config.dataset.has_sub_nodes:
      ## Logic is made similar to the sampling part
      ## Instead of multinomial, here we take argmax to return most probable alpha
      ## Then, instead of choosing whether an edge exists, we take its sigmoid
      ## and use the sigmoid to scale the predicted subnodes, if edge should not be there
      ## It will lead to higher loss!\
      ## Furthermore, at sampling, each node is sampled from different alpha (mixture), this 
      ## logic implemented below, does the same
      grp_subgraph = groupidx(subgraph_idx) ## This had to be done to handle batch size and nodes properly
      prob = torch.zeros((log_alpha.shape[0],1), dtype=torch.float32).to(self.device)
      for grp in grp_subgraph:
        log_alpha_subsample = log_alpha[grp]
        prob_alpha = F.softmax(log_alpha_subsample.mean(dim=0), -1)
        alpha_idx = torch.argmax(prob_alpha, keepdim=True).reshape(-1)

        log_theta_subsample = log_theta[grp]
        prob[grp] = torch.sigmoid(log_theta_subsample[:, alpha_idx])

      scaled_output_subnodes = output_subnodes * prob
      criterion = nn.MSELoss()
      subnode_loss = criterion(scaled_output_subnodes, subnode_labels)
      subnode_loss = subnode_loss
    
    return adj_loss, embed_loss, subnode_loss

  def get_truncated_graphs(self, A, batch_size, N_max, num_nodes, node_embed_out, 
                           subnode_coords_out, theta_ret):
    '''
    The following functions uses num_nodes to return part of graph before stop_node
    '''
    ## Default return values
    A_list, node_embed_list, subnode_coords_list, theta_list = None, None, None, None

    A_list = [
          A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
      ]
    if self.config.dataset.has_node_feat:
      node_embed_list = [
        node_embed_out[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
      ]
    if self.config.dataset.has_sub_nodes:
      subnode_coords_reshape = subnode_coords_out.reshape(batch_size, N_max,N_max, -1)
      subnode_coords_list = [
        subnode_coords_reshape[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
      ]
    if self.config.test.animated_vis:
      theta_list = [
        theta_ret[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
      ]

    return A_list, node_embed_list, subnode_coords_list, theta_list

  def forward(self, input_dict, graph_completion = False):
    """
      B: batch size
      N: number of rows/columns in mini-batch
      N_max: number of max number of rows/columns
      M: number of augmented edges in mini-batch
      H: input dimension of GNN
      K: block size
      E: number of edges in mini-batch
      S: stride
      C: number of canonical orderings
      D: number of mixture Bernoulli

      Args:
        A_pad: B X C X N_max X N_max, padded adjacency matrix
        node_idx_gnn: M X 2, node indices of augmented edges
        node_idx_feat: N X 1, node indices of subgraphs for indexing from feature
                      (0 indicates indexing from 0-th row of feature which is
                        always zero and corresponds to newly generated nodes)
        att_idx: N X 1, one-hot encoding of newly generated nodes
                      (0 indicates existing nodes, 1-D indicates new nodes in
                        the to-be-generated block)
        subgraph_idx: E X 1, indices corresponding to augmented edges
                      (representing which subgraph in mini-batch the augmented
                      edge belongs to)
        edges: E X 2, edge as [incoming node index, outgoing node index]
        label: E X 1, binary label of augmented edges
        num_nodes_pmf: N_max, empirical probability mass function of number of nodes

      Returns:
        loss                        if training
        list of adjacency matrices  else
    """
    if isinstance(input_dict, tuple):
      input_dict = input_dict[0]
    
    is_sampling, batch_size, A_pad, node_embed_pad, node_embed_idx_gnn, label_embed \
    , subnode_coords, subnode_labels, node_idx_gnn, node_idx_feat, att_idx, subgraph_idx \
    , edges, label, num_nodes_pmf, subgraph_idx_base = self.dict2var(input_dict)

    N_max = self.max_num_nodes

    if not is_sampling:
      B, _, N, _ = A_pad.shape

      ### compute adj loss
      log_theta, log_alpha, embed_log_pis, embed_mv, output_subnodes  = self._inference(
        A_pad=A_pad,
        edges=edges,
        node_idx_gnn=node_idx_gnn,
        node_idx_feat=node_idx_feat,
        att_idx=att_idx,
        node_embed_pad = node_embed_pad,
        node_embed_idx_gnn = node_embed_idx_gnn,
        subnode_coords = subnode_coords)

      num_edges = log_theta.shape[0]

      adj_loss, embed_loss, subnode_loss = self.get_losses(log_theta, log_alpha, subgraph_idx, 
                 subgraph_idx_base, label, embed_log_pis, embed_mv, label_embed,output_subnodes,
                 subnode_labels)

      return adj_loss, embed_loss, subnode_loss

    else:

      A, node_embed_out, subnode_coords_out, theta_ret = self._sampling(batch_size)

      if not self.config.dataset.has_stop_node:
        ### sample number of nodes
        num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(self.device)
        num_nodes = torch.multinomial(
            num_nodes_pmf, batch_size, replacement=True) + 1  # shape B X 1
      else:
        ### We chose num_nodes on the basis of stop node
        num_nodes = []
        for cur_A in A:
          cmp_arr = torch.zeros((1,cur_A.shape[0]), dtype=torch.float32).to(self.device)
          # cmp_arr[0, -1:] = 0.0
          has_end = False
          for ii in range(len(cur_A)):
            if ii <= 3:
              continue # We assume graphs to have more than 3 nodes at least
            ## The stop node will be connected to all previous nodes
            ## So, the adjacency should have a row of 1's for previous nodes followed by 0's
            cmp_arr[0, :ii] = 1.0
            adj_lower_row = torch.zeros_like(cur_A[ii])
            adj_lower_row[:ii] = cur_A[ii, :ii]
            check_close =  torch.isclose(cmp_arr[0], adj_lower_row)
            if check_close.all():
              has_end = True
              num_nodes += [ii]
              break
          if not has_end:
            num_nodes += [len(cur_A)-1]
            
      A_list, node_embed_list, subnode_coords_list, \
        theta_list = self.get_truncated_graphs(A, batch_size, N_max, num_nodes, 
                                               node_embed_out, subnode_coords_out, theta_ret)

      return A_list, node_embed_list, subnode_coords_list, theta_list

def mixture_bernoulli_loss(label, log_theta, log_alpha, adj_loss_func,
                           subgraph_idx, subgraph_idx_base, num_canonical_order, 
                           sum_order_log_prob=False, return_neg_log_prob=False, reduction="mean"):
  """
    Compute likelihood for mixture of Bernoulli model

    Args:
      label: E X 1, see comments above
      log_theta: E X D, see comments above
      log_alpha: E X D, see comments above
      adj_loss_func: BCE loss
      subgraph_idx: E X 1, see comments above
      subgraph_idx_base: B+1, cumulative # of edges in the subgraphs associated with each batch
      num_canonical_order: int, number of node orderings considered
      sum_order_log_prob: boolean, if True sum the log prob of orderings instead of taking logsumexp 
        i.e. log p(G, pi_1) + log p(G, pi_2) instead of log [p(G, pi_1) + p(G, pi_2)]
        This is equivalent to the original GRAN loss.
      return_neg_log_prob: boolean, if True also return neg log prob
      reduction: string, type of reduction on batch dimension ("mean", "sum", "none")

    Returns:
      loss (and potentially neg log prob)
  """

  num_subgraph = subgraph_idx_base[-1] # == subgraph_idx.max() + 1
  B = subgraph_idx_base.shape[0] - 1
  C = num_canonical_order
  E = log_theta.shape[0]
  K = log_theta.shape[1]
  assert E % C == 0
  adj_loss = torch.stack(
      [adj_loss_func(log_theta[:, kk], label) for kk in range(K)], dim=1)

  const = torch.zeros(num_subgraph).to(label.device) # S
  const = const.scatter_add(0, subgraph_idx,
                            torch.ones_like(subgraph_idx).float())

  reduce_adj_loss = torch.zeros(num_subgraph, K).to(label.device)
  reduce_adj_loss = reduce_adj_loss.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), adj_loss)

  reduce_log_alpha = torch.zeros(num_subgraph, K).to(label.device)
  reduce_log_alpha = reduce_log_alpha.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), log_alpha)
  reduce_log_alpha = reduce_log_alpha / const.view(-1, 1)
  reduce_log_alpha = F.log_softmax(reduce_log_alpha, -1)

  log_prob = -reduce_adj_loss + reduce_log_alpha
  log_prob = torch.logsumexp(log_prob, dim=1) # S, K

  bc_log_prob = torch.zeros([B*C]).to(label.device) # B*C
  bc_idx = torch.arange(B*C).to(label.device) # B*C
  bc_const = torch.zeros(B*C).to(label.device)
  bc_size = (subgraph_idx_base[1:] - subgraph_idx_base[:-1]) // C # B
  bc_size = torch.repeat_interleave(bc_size, C) # B*C
  bc_idx = torch.repeat_interleave(bc_idx, bc_size) # S
  bc_log_prob = bc_log_prob.scatter_add(0, bc_idx, log_prob)
  # loss must be normalized for numerical stability
  bc_const = bc_const.scatter_add(0, bc_idx, const)
  bc_loss = (bc_log_prob / bc_const)

  bc_log_prob = bc_log_prob.reshape(B,C)
  bc_loss = bc_loss.reshape(B,C)
  if sum_order_log_prob:
    b_log_prob = torch.sum(bc_log_prob, dim=1)
    b_loss = torch.sum(bc_loss, dim=1)
  else:
    b_log_prob = torch.logsumexp(bc_log_prob, dim=1)
    b_loss = torch.logsumexp(bc_loss, dim=1)

  # probability calculation was for lower-triangular edges
  # must be squared to get probability for entire graph
  b_neg_log_prob = -2*b_log_prob
  b_loss = -b_loss
  
  if reduction == "mean":
    neg_log_prob = b_neg_log_prob.mean()
    loss = b_loss.mean()
  elif reduction == "sum":
    neg_log_prob = b_neg_log_prob.sum()
    loss = b_loss.sum()
  else:
    assert reduction == "none"
    neg_log_prob = b_neg_log_prob
    loss = b_loss
  
  if return_neg_log_prob:
    return loss, neg_log_prob
  else:
    return loss
  
def groupidx(in_tensor):
  '''
  This function will help us group subgraphs so that we can calculate alpha and theta correctly
  '''
  subgraph_list = in_tensor.cpu().numpy().tolist()
  result=[] # Output List
  for key,group in groupby(enumerate(subgraph_list),lambda x:x[1]):
      result.append([i for i,k in group])
  return result