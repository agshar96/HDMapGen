from __future__ import (division, print_function)
import os
import time
import networkx as nx
import numpy as np
import copy
import pickle
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed as distributed

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.data_helper import *
from utils.eval_helper import *
from utils.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv
from utils.vis_helper import draw_graph_list, draw_graph_list_separate, draw_graph_list_embed, draw_graph_nodes_list, draw_animated_plot, draw_graph_subnode_list, draw_animated_subnodes
from utils.data_parallel import DataParallel
from utils.lsg_helper import large_scale_vis


try:
  ###
  # workaround for solving the issue of multi-worker
  # https://github.com/pytorch/pytorch/issues/973
  import resource
  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
  ###
except:
  pass

logger = get_logger('exp_logger')
__all__ = ['GranRunner', 'compute_edge_ratio', 'get_graph']

NPR = np.random.RandomState(seed=1234)

def compute_edge_ratio(G_list):
  num_edges_max, num_edges = .0, .0
  for gg in G_list:
    num_nodes = gg.number_of_nodes()
    num_edges += gg.number_of_edges()
    num_edges_max += num_nodes**2

  ratio = (num_edges_max - num_edges) / num_edges
  return ratio


def get_graph(adj):
  """ get a graph from zero-padded adj """
  # remove all zeros rows and columns
  adj = adj[~np.all(adj == 0, axis=1)]
  adj = adj[:, ~np.all(adj == 0, axis=0)]
  adj = np.asmatrix(adj)
  G = nx.from_numpy_array(adj)
  return G

def get_graph_embed(adj, node_embed):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    ## UNCOMMENT IN MAIN CODE
    # remove all zeros rows and columns
    # adj = adj[~np.all(adj == 0, axis=1)]
    # adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_array(adj)
    feature_dict = {}
    for node_id, feature_vector in enumerate(node_embed):
        feature_dict[node_id] = {'features': feature_vector}
    
    nx.set_node_attributes(G, feature_dict)
    return G

def get_graph_subnodes(adj, node_embed, subnode_coords):

    adj = np.asmatrix(adj)
    G = nx.from_numpy_array(adj)
    # for row in range(adj.shape[0]):
    #   for col in range(row+1):
    #     if adj[row, col] == 1:
    #       G[row][col].update({'subnodes': subnode_coords[u,v]})
    #       test = subnode_coords[row,col, :]
    #       print('tested')

    feature_dict = {}
    for node_id, feature_vector in enumerate(node_embed):
        feature_dict[node_id] = {'features': feature_vector}
    
    edges = G.edges()
    for u,v in edges:
        if v <= u:
          subnodes = subnode_coords[u,v].reshape(-1,2)
          G[u][v].update({'subnodes': subnodes})
        elif v > u:
          subnodes = subnode_coords[v,u].reshape(-1,2)
          subnodes = np.flip(subnodes, axis=0)
          G[u][v].update({'subnodes': subnodes})
    
    nx.set_node_attributes(G, feature_dict)
    return G

class GranRunner(object):

  def configure_optimizer(self, train_conf, params):
    if train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
          params,
          lr=train_conf.lr,
          momentum=train_conf.momentum,
          weight_decay=train_conf.wd)
    elif train_conf.optimizer == 'Adam':
      optimizer = optim.Adam(params, lr=train_conf.lr, weight_decay=train_conf.wd)
    elif train_conf.optimizer == 'AdamW':
      optimizer = optim.AdamW(params, lr=train_conf.lr)
    else:
      raise ValueError("Non-supported optimizer!")

    early_stop = EarlyStopper([0.0], win_size=100, is_decrease=False)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_conf.lr_decay_epoch,
        gamma=train_conf.lr_decay)
    
    return optimizer, lr_scheduler

  def get_batch_fwd(self, batch_data, ff):
    '''
    This function assigns data batches to specific gpus.
    Since the code was adapted from GRAN thus, this logic had to be integrated
    '''
    batch_fwd = []
          
    for dd, gpu_id in enumerate(self.gpus):
      data = {}
      data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)
      if self.config.dataset.has_node_feat:
        data['node_embed'] = batch_data[dd][ff]['node_embed'].pin_memory().to(gpu_id, non_blocking=True)
        data['node_embed_idx_gnn'] = batch_data[dd][ff]['node_embed_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
        data['label_embed'] = batch_data[dd][ff]['label_embed'].pin_memory().to(gpu_id, non_blocking=True)
      if self.config.dataset.has_sub_nodes:
        data['subnode_coords'] = batch_data[dd][ff]['subnode_coords'].pin_memory().to(gpu_id, non_blocking=True)
        data['subnode_labels'] = batch_data[dd][ff]['subnode_labels'].pin_memory().to(gpu_id, non_blocking=True)
      data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
      data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
      data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
      data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
      data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
      data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
      data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].pin_memory().to(gpu_id, non_blocking=True)
      batch_fwd.append((data,))
    return batch_fwd

  def log_data(self, results, iter_count, train_loss, train_adj_loss, train_embed_loss, 
               train_subnode_loss, epoch):
    '''
    If a feature is not enabled the corresponding loss is 0
    '''
    # Tensorboard logging
    self.writer.add_scalar('train_loss', train_loss, iter_count)
    results['train_loss'] += [train_loss]
    results['train_step'] += [iter_count]
    results['train_subnode_loss'] += [train_subnode_loss]
    self.writer.add_scalar('train_subnode_loss', train_subnode_loss, iter_count)
    results['train_adj_loss'] += [train_adj_loss]
    results['train_embed_loss'] += [train_embed_loss]
    self.writer.add_scalar('train_adj_loss', train_adj_loss, iter_count)
    self.writer.add_scalar('train_embed_loss', train_embed_loss, iter_count)

    if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
      logger.info("NLL Loss @ epoch {:04d} iteration {:08d} = {}, train_adj_loss = {}, train_embed_loss = {}, train_subnode_loss = {}".format(
        epoch + 1, iter_count, train_loss, train_adj_loss, train_embed_loss, train_subnode_loss))

    
  def vis_gen_graphs(self, num_row, num_col, graph_gen_no_embed, graphs_gen, theta_pred):
    '''
    Function to visualize generated graphs, to be used for debugging.
    '''
    if self.config.dataset.has_sub_nodes:
      draw_graph_subnode_list(graphs_gen[:self.num_vis], num_row, num_col, fname="test_graphs_subnodes_1")
      if self.config.test.animated_vis:
        draw_animated_subnodes(graphs_gen[:self.num_vis], theta_pred, num_graphs=self.config.test.num_animations, 
                              fname = 'graph_subnode_animation',
                              x_lim=(-2,2), y_lim=(-2,2))

    ### If graph has subnode it will have embedding therefore another if
    if self.config.dataset.has_node_feat:
      if self.config.test.animated_vis:
        draw_animated_plot(graphs_gen[:self.num_vis], theta_pred, num_graphs=self.config.test.num_animations, fname = 'graph_embed_animation',
                            x_lim=(-2,2), y_lim=(-2,2))
      draw_graph_list_embed(graphs_gen[:self.num_vis], num_row, num_col, fname="test_graphs_1")
      draw_graph_nodes_list(graphs_gen[:self.num_vis], num_row, num_col, fname="test_nodes_1")

    ### Base case with no embedding
    draw_graph_list(graph_gen_no_embed[:self.num_vis], num_row, num_col, fname="test_no_embed", layout='spring')

  def vis_train_graphs(self, num_row, num_col):
    '''
    Function to visualize some graphs from training data set
    '''
    save_name = os.path.join(self.config.save_dir, 'train_graphs.png')

    if self.config.dataset.has_node_feat:
      draw_graph_list_embed(self.graphs_train[:self.num_vis], num_row, num_col, fname="train_graphs")
    else:
      draw_graph_list(
          self.graphs_train[:self.num_vis],
          num_row,
          num_col,
          fname='train_graphs.png',
          layout='spring')

  def __init__(self, config):
    self.config = config
    self.seed = config.seed
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.device = config.device
    self.writer = SummaryWriter(config.save_dir)
    self.is_vis = config.test.is_vis
    self.num_vis = config.test.num_vis
    self.vis_num_row = config.test.vis_num_row
    self.num_gpus = len(self.gpus)
    self.is_shuffle = False

    assert self.use_gpu == True

    if self.train_conf.is_resume:
      self.config.save_dir = self.train_conf.resume_dir

    ### load graphs
    if not hasattr(config.dataset, "is_noisy"):
      config.dataset.is_noisy = False
      config.dataset.noise_std = 1.0
    self.graphs = create_graphs(config.dataset.name)
    
    self.train_ratio = config.dataset.train_ratio
    self.dev_ratio = config.dataset.dev_ratio
    self.block_size = config.model.block_size
    self.stride = config.model.sample_stride
    self.num_graphs = len(self.graphs)
    self.num_train = int(float(self.num_graphs) * self.train_ratio)
    self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
    self.num_test_gt = self.num_graphs - self.num_train
    self.num_test_gen = config.test.num_test_gen

    logger.info('Train/val/test = {}/{}/{}'.format(self.num_train, self.num_dev,
                                                  self.num_test_gt))

    ### shuffle all graphs
    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      self.npr.shuffle(self.graphs)

    self.graphs_train = self.graphs[:self.num_train]
    self.graphs_dev = self.graphs[:self.num_dev]
    self.graphs_test = self.graphs[self.num_train:]
    
    # draw_graph_subnode_list(self.graphs_train[6:8], 1, 2)

    self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs_train)
    logger.info('No Edges vs. Edges in training set = {}'.format(
        self.config.dataset.sparse_ratio))

    self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])[1:]  # remove 'zero occurrence'    
    self.max_num_nodes = len(self.num_nodes_pmf_train)
    self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()
    
    ### save split for benchmarking
    if config.dataset.is_save_split:      
      base_path = os.path.join(config.dataset.data_path, 'save_split')
      if not os.path.exists(base_path):
        os.makedirs(base_path)
      
      save_graph_list(
          self.graphs_train,
          os.path.join(base_path, '{}_train.p'.format(config.dataset.name)))
      save_graph_list(
          self.graphs_dev,
          os.path.join(base_path, '{}_dev.p'.format(config.dataset.name)))
      save_graph_list(
          self.graphs_test,
          os.path.join(base_path, '{}_test.p'.format(config.dataset.name)))

  def train(self):
    ### create data loader
    train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_train, tag='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)

    if self.use_gpu:
      model = DataParallel(model, device_ids=self.gpus).to(self.device)

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer, lr_scheduler = self.configure_optimizer(self.train_conf, params)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    resume_epoch = 0
    if self.train_conf.is_resume:
      model_file = os.path.join(self.train_conf.resume_dir,
                                self.train_conf.resume_model)
      load_model(
          model.module if self.use_gpu else model,
          model_file,
          self.device,
          optimizer=optimizer,
          scheduler=lr_scheduler)
      resume_epoch = self.train_conf.resume_epoch
      ## Workaround for a bug which appears only during model resume :(
      optimizer.param_groups[0]['capturable'] = True

    # Training Loop
    iter_count = 0    
    results = defaultdict(list)
    for epoch in range(resume_epoch, self.train_conf.max_epoch):
      model.train()
      lr_scheduler.step()
      train_iterator = train_loader.__iter__()

      for inner_iter in range(len(train_loader) // self.num_gpus):
        optimizer.zero_grad()

        batch_data = []
        if self.use_gpu:
          for _ in self.gpus:
            data = train_iterator.next()
            batch_data.append(data)
            iter_count += 1
        
        avg_train_loss = .0
        # In case if subnode, or embedding are turned off, those losses will be logged as zeros
        avg_train_subnode_loss = 0
        avg_train_adj_loss = 0
        avg_train_embed_loss = 0

        for ff in range(self.dataset_conf.num_fwd_pass):
          batch_fwd = self.get_batch_fwd(batch_data, ff)

          if batch_fwd:
            train_adj_loss, train_embed_loss, train_subnode_loss = model(*batch_fwd)
            # train_embed_loss = 100 * train_adj_loss # artificially boosting the loss to see what happens
            train_loss = (train_adj_loss + train_embed_loss + train_subnode_loss).mean()
            train_loss = train_loss.to(torch.float32)
            avg_train_loss += train_loss 
            avg_train_adj_loss += train_adj_loss.mean()
            if self.config.dataset.has_node_feat:
              avg_train_embed_loss += train_embed_loss.mean()
            if self.config.dataset.has_sub_nodes:
              avg_train_subnode_loss += train_subnode_loss.mean()       

            # assign gradient
            train_loss.backward()
        
        # clip_grad_norm_(model.parameters(), 5.0e-0)
        optimizer.step()
        # Calculate averages and get data for printing and logging
        avg_train_loss /= float(self.dataset_conf.num_fwd_pass)
        avg_train_adj_loss /= float(self.dataset_conf.num_fwd_pass)
        train_adj_loss = float(avg_train_adj_loss.data.cpu().numpy())
        if self.config.dataset.has_node_feat:
          avg_train_embed_loss /= float(self.dataset_conf.num_fwd_pass)
          train_embed_loss = float(avg_train_embed_loss.data.cpu().numpy())
        if self.config.dataset.has_sub_nodes:
          avg_train_subnode_loss /= float(self.dataset_conf.num_fwd_pass)
          train_subnode_loss = float(avg_train_subnode_loss.data.cpu().numpy())
        
        # reduce
        train_loss = float(avg_train_loss.data.cpu().numpy())

        self.log_data(results, iter_count, train_loss, train_adj_loss, train_embed_loss, 
               train_subnode_loss, epoch)

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0 and (epoch + 1) > self.train_conf.valid_epoch:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)
    
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    
    return 1

  def test(self):

    if not hasattr(self.config.test, "animated_vis"):
      self.config.test.animated_vis = False

    ### load model
    model = eval(self.model_conf.name)(self.config)
    model_file = os.path.join(self.test_conf.test_model_dir, self.test_conf.test_model_name)
    load_model(model, model_file, self.device)

    if self.use_gpu:
      model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)

    model.eval()

    ### Initialize lists to be used for vis
    A_pred = []
    num_nodes_pred = []
    subnode_pred = []
    node_embed_pred = []
    theta_pred = []
    num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))

    gen_run_time = []
    for ii in tqdm(range(num_test_batch)):
      with torch.no_grad():        
        start_time = time.time()
        input_dict = {}
        input_dict['is_sampling']=True
        input_dict['batch_size']=self.test_conf.batch_size
        input_dict['num_nodes_pmf']=self.num_nodes_pmf_train

        ## Generate the graphs from the model
        A_tmp, node_embed_tmp, subnode_tmp, theta_tmp = model(input_dict)

        gen_run_time += [time.time() - start_time]

        ## Convert from tensors to numpy for vis and saving
        A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
        num_nodes_pred += [aa.shape[0] for aa in A_tmp]
        if self.config.dataset.has_sub_nodes:
          subnode_pred += [aa.data.cpu().numpy() for aa in subnode_tmp]
        if self.config.dataset.has_node_feat:
          node_embed_pred += [aa.data.cpu().numpy() for aa in node_embed_tmp]
        if self.config.test.animated_vis:
          theta_pred += [aa.data.cpu().numpy() for aa in theta_tmp]

    logger.info('Average test time per mini-batch = {}'.format(
        np.mean(gen_run_time)))
    
    ### Create networkx graphs from returned information
    if self.config.dataset.has_sub_nodes:
      graphs_gen = [get_graph_subnodes(A_pred[ii], node_embed_pred[ii], subnode_pred[ii]) for ii in range(len(A_pred))]
      graph_gen_no_embed = [get_graph(aa) for aa in A_pred]
    elif self.config.dataset.has_node_feat:
      graphs_gen = [get_graph_embed(A_pred[ii], node_embed_pred[ii]) for ii in range(len(A_pred))]
      graph_gen_no_embed = [get_graph(aa) for aa in A_pred]
    else:
      graphs_gen = None
      theta_pred = None # No animation for base graphs
      graph_gen_no_embed = [get_graph(aa) for aa in A_pred]
    
    ### This settings is used to save all visualizations and graphs at once
    ### This needs subnodes to be enabled as large scale visualization make sense for the final model.
    if self.config.test.large_scale_gen:
      large_scale_vis(graphs_gen, self.config)

    ### Visualize Generated Graphs
    ## Use this setting for debugging, for visualizing large number of graphs use large_scale_gen
    if self.is_vis:
      num_col = self.vis_num_row
      num_row = int(np.ceil(self.num_vis / num_col))

      self.vis_gen_graphs(num_row, num_col, graph_gen_no_embed, graphs_gen, theta_pred)

      self.vis_train_graphs(num_row, num_col)