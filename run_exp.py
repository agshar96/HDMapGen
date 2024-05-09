import sys
import torch
import traceback
from runner import *
from utils.logger import init_logger
from utils.arg_helper import init_args_config
torch.set_printoptions(profile='full')


def main():
  # Read arguments and store in config
  args, config = init_args_config()

  # log info
  logger = init_logger(config, args)

  # Run the experiment
  # args.test = True #Comment while training
  try:
    runner = eval(config.runner)(config)
    if not args.test:
      runner.train()
    else:
      runner.test()
  except:
    logger.error(traceback.format_exc())

  sys.exit(0)


if __name__ == "__main__":
  main()