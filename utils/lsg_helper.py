'''
It contains the functions for large scale graph visulization
'''
import numpy as np
import os
from utils.vis_helper import draw_graph_subnode_list, draw_graph_list_embed, draw_graph_nodes_list
import pickle

def draw_graphs(graphs_gen, num_row, num_col, num_files, save_prefix):
    per_file = num_col * num_row
    for file_num in range(num_files):
        cur_file_name = save_prefix + "_subnode_" + str(file_num) + ".png"

        cur_graphs = graphs_gen[per_file * file_num : per_file * (file_num + 1)]
        draw_graph_subnode_list(cur_graphs, num_row, num_col, fname= cur_file_name)

        cur_file_name = save_prefix + "_embed_" + str(file_num) + ".png"
        draw_graph_list_embed(cur_graphs, num_row, num_col, fname= cur_file_name)

        cur_file_name = save_prefix + "_nodes_" + str(file_num) + ".png"
        draw_graph_nodes_list(cur_graphs, num_row, num_col, fname= cur_file_name)

def save_graph_list(graphs_gen, save_prefix):
    filename = save_prefix + "_graph_list.p"
    outfile = open(filename,'wb')
    pickle.dump(graphs_gen , outfile)
    outfile.close()

def large_scale_vis(graphs_gen ,config):
    num_row = config.test.vis_num_row
    num_col = int(np.ceil(config.test.num_vis / num_row))
    num_files = int(np.ceil(len(graphs_gen) / (num_row * num_col) ))
    save_name = os.path.join(config.save_dir, 'graph_gen.png')

    save_graph_list(graphs_gen, save_prefix=save_name[:-4])
    draw_graphs(graphs_gen, num_row, num_col, num_files,
                             save_prefix=save_name[:-4])
    
    