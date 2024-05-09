from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt

import gzip
import pickle


@dataclass
class HDMapGen:

    nodes: npt.NDArray[np.float32]  # shape (num_nodes, 2)
    connections: npt.NDArray[np.int16]  # shape (num_connections, 2)
    subnodes: npt.NDArray[np.float32]  # shape (num_connections, 20, 2)


def load_gz_dirs(gz_dirs: List[Path]) -> List[HDMapGen]:
    features: List[HDMapGen] = []
    for gz_dir in gz_dirs:
        if gz_dir.exists():
            with gzip.open(gz_dir, "rb") as f:
                data = pickle.load(f)
        else:
            continue

        features.append(
            HDMapGen(
                nodes=data["nodes"],
                connections=data["connections"],
                subnodes=data["subnodes"],
            )
        )

    return features

def find_gz_dirs(root_path: Path, file_name: str = "hdmapgen.gz") -> List[Path]:
    """Find files in a directory structure that end with a specified extension"""

    gz_dirs = []
    for log_path in root_path.iterdir():

        if "metadata" in str(log_path):
            continue

        for scenario_path in log_path.iterdir():
            for token_path in scenario_path.iterdir():
                gz_dirs.append(token_path / file_name)

    return gz_dirs


def get_data_nuplan(root_path = Path('/home/hiwi-1/Documents/GRAN_Embedded/HDMapGen/hdmapgen')):
    features = load_gz_dirs(find_gz_dirs(root_path))
    return features

# output = get_data_nuplan()
# max_nodes = -1
# min_nodes = 1000
# for item in output:
#     tmp_nodes = item.nodes.shape[0]
#     if tmp_nodes > max_nodes:
#         max_nodes = tmp_nodes
#     if tmp_nodes < min_nodes:
#         min_nodes = tmp_nodes

# print("Max nodes: ", max_nodes, " Min nodes: ", min_nodes)