import os
import yaml
import time
import argparse
from easydict import EasyDict as edict
from utils.config_resolver import resolve_config
import numpy as np
import torch


def parse_arguments():
  parser = argparse.ArgumentParser(
      description="Running Experiments of Graph Generation")
  parser.add_argument(
      '-c',
      '--config_file',
      type=str,
      default="config/gran_subnode.yaml",
      # required=True,
      help="Path of config file")
  parser.add_argument(
      '-l',
      '--log_level',
      type=str,
      default='INFO',
      help="Logging Level, \
        DEBUG, \
        INFO, \
        WARNING, \
        ERROR, \
        CRITICAL")
  parser.add_argument('-m', '--comment', help="Experiment comment")
  parser.add_argument('-t', '--test', help="Test model", action='store_true')
  args = parser.parse_args()

  return args


def get_config(config_file, exp_dir=None, is_test=False):
  """ Construct and snapshot hyper parameters """
  config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
  # config = edict(yaml.load(open(config_file, 'r')))

  # create hyper parameters
  config.run_id = str(os.getpid())
  config.exp_name = '_'.join([
      config.model.name, config.dataset.name,
      time.strftime('%Y-%b-%d-%H-%M-%S'), config.run_id
  ])

  if exp_dir is not None:
    config.exp_dir = exp_dir
  
  if config.train.is_resume and not is_test:
    config.save_dir = config.train.resume_dir
    save_name = os.path.join(config.save_dir, 'config_resume_{}.yaml'.format(config.run_id))  
  else:    
    config.save_dir = os.path.join(config.exp_dir, config.exp_name)
    save_name = os.path.join(config.save_dir, 'config.yaml')

  # snapshot hyperparameters
  mkdir(config.exp_dir)
  mkdir(config.save_dir)

  yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

  return config


def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj


def mkdir(folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)

def init_args_config():

  args = parse_arguments()
  config = get_config(args.config_file, is_test=(args.test))
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  config.use_gpu = config.use_gpu and torch.cuda.is_available()

  config = resolve_config(config)
  return args, config

