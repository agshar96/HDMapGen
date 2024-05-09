import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.data_helper import *
import networkx as nx

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def lower_triangular_matrix(row_num):
    if row_num <= 0:
        raise ValueError("Number of rows must be a positive integer.")

    matrix = np.zeros((row_num, row_num))

    for i in range(row_num):
        row_sum = np.random.rand(i + 1)
        row_sum /= row_sum.sum()
        matrix[i, :i + 1] = row_sum

    return matrix

def animate_embed_graph():
    graphs = create_graphs("subnode")

    graphs_train = graphs[:5]

    graph_to_plot = graphs_train[0]

    node_embed_output = list(graph_to_plot.nodes(data=True))
    node_embed_output = np.array([item[1]['features'] for item in node_embed_output])
    adj = np.asarray(nx.to_numpy_array(graph_to_plot))
    lower_adj = np.tril(adj)
    theta_mat = lower_triangular_matrix(node_embed_output.shape[0])
    edges = graph_to_plot.edges()
    edge_subnode_dict = {}
    for u,v in edges:
        edge_subnode_dict[(u,v)] = graph_to_plot[u][v]['subnodes'].numpy()

    fig, ax = plt.subplots()
    sc = ax.scatter([], [], color='red', s=15)
    sbnds = ax.scatter([], [], color='blue', s=5) # For subnodes
    lines = LineCollection([], cmap='Greys', norm=plt.Normalize(0, 1))

    #### CODE FOR COLOR MAP
    # Create a ScalarMappable to display the colormap
    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(cmap='Greys', norm=norm)
    sm.set_array([])  # dummy empty array to satisfy the ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Alpha Values')



    ax.add_collection(lines)
    x_lim = (-1,4)
    ax.set_xlim(x_lim)
    ax.set_ylim(x_lim)

    prev_alpha = None
    def update(frame):
        
        nonlocal prev_alpha
        if frame % 3 == 0:
            cur_frame = int(frame/3)
            sc.set_offsets(np.column_stack((node_embed_output[:cur_frame+1,0], 
                                            node_embed_output[:cur_frame+1,1])))
            

            if cur_frame != 0:
                segments = []
                i=0
                while i < cur_frame:
                    cur_line = np.vstack((node_embed_output[i],node_embed_output[cur_frame]))
                    segments.append(cur_line)
                    i+=1
                prev_segments = lines.get_segments()
                segments = np.stack(segments)
                theta = theta_mat[cur_frame,:cur_frame]
                if prev_segments:
                    prev_segments = np.stack(prev_segments)
                    segments = np.vstack((prev_segments, segments))
                    theta = np.concatenate((prev_alpha, theta))
                
                lines.set_segments(segments)
                # colors = sm.to_rgba(theta)
                # lines.set_color(colors)
                lines.set_alpha(theta)

        elif frame % 3 == 1:
            # pass
            if frame != 1:
                cur_frame = int((frame - 1)/3)
                alpha = lower_adj[cur_frame,:cur_frame]
                if prev_alpha is not None:
                    alpha = np.concatenate((prev_alpha, alpha))
                cur_segments = np.stack(lines.get_segments())
                idxs = np.where(alpha == 1)[0].tolist()
                cur_segments = cur_segments[idxs]
                alpha_ones = alpha[idxs]
                # if frame == 2*node_embed_output.shape[0] -1 :
                # lines.set_array(alpha)
                lines.set_segments(cur_segments)
                # colors = sm.to_rgba(alpha_ones)
                # lines.set_color(colors)
                lines.set_alpha(alpha_ones)
                prev_alpha = alpha_ones
        
        else:

            if frame != 2:
                prev_subnodes = sbnds.get_offsets().data
                cur_frame = int((frame - 2)/3)
                adjs = lower_adj[cur_frame,:cur_frame]
                for idx, item in enumerate(adjs):
                    if item == 1:
                        new_subnodes = np.flip(edge_subnode_dict[(idx, cur_frame)], axis=0)
                        prev_subnodes = np.concatenate((prev_subnodes, new_subnodes), axis = 0)

                sbnds.set_offsets(prev_subnodes)

        return sc,
        
    animation = FuncAnimation(fig, update, frames=3*node_embed_output.shape[0], interval=500, blit=True)

    animation.save('basic_animation.mp4', fps=1)
# line, = ax.plot([], [], linestyle='-', color='b')
animate_embed_graph()


# Your data
# x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = np.array([12, 56, 78, 90, 23, 11, 67, 89, 12, 34, 56])

# # Create a figure and axis
# fig, ax = plt.subplots()
# sc = ax.scatter([], [])

# # Set axis limits
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 100)

# # Update function for animation
# def update(frame):
#     test1 = x[:frame+1]
#     test2 = y[:frame+1]
#     test3 = np.column_stack((x[:frame+1], y[:frame+1]))
#     sc.set_offsets(np.column_stack((x[:frame+1], y[:frame+1])))
#     return sc,

# # Create the animation
# animation = FuncAnimation(fig, update, frames=len(x), interval=500, blit=True)

# # Show the plot
# animation.save('basic_animation.mp4', fps=1)