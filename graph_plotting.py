import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    from data_creation import create_random_network
from models.training import easyTrainer, weights_to_dataset
from models.nODE import nODE, make_nODE_from_parameters
import scipy
import torch


alpha_const = 0.6


def plot_circle(center=None, radius=1., color=None, linewidth=1.):
    if center is None:
        center = [0, 0]
    t = np.linspace(0, 2 * np.pi, 60)
    if color:
        plt.plot(center[0] + radius * np.sin(t), center[1] + radius * np.cos(t), color=color, linewidth=linewidth)
    else:
        plt.plot(center[0] + radius * np.sin(t), center[1] + radius * np.cos(t), linewidth=linewidth)


def plot_arch(p1, p2, color=None, linewidth=7., alpha=alpha_const):
    # assume p1 and p2 are on the boundary of the circle
    if all(p1 == p2):
        self_edge(p1, linewidth=linewidth, color=color, alpha=alpha)
        return
    center = (p1 + p2) / 2
    direction = p1 - center
    radius = np.linalg.norm(direction)
    direction = direction / radius
    t_plot = np.linspace(.7, np.pi - .7, 60)
    perpendicular = np.array([direction[1], -direction[0]])
    perturbation_x = center[0] + np.cos(t_plot) * radius * direction[0] + 0.15 * np.sin(t_plot) * perpendicular[
        0] * radius
    perturbation_y = center[1] + np.cos(t_plot) * radius * direction[1] + 0.15 * np.sin(t_plot) * perpendicular[
        1] * radius

    middle = center + np.cos(np.pi * 2 / 3) * direction * radius + 0.15 * np.sin(np.pi * 2 / 3) * perpendicular * radius
    arrow = 0.1 * np.array([-perpendicular + direction, 0 * direction, perpendicular + direction]) + middle

    if color is not None:
        plt.plot(perturbation_x, perturbation_y, linewidth=linewidth, color=color, alpha=alpha)
        plt.plot(arrow[:, 0], arrow[:, 1], linewidth=linewidth, color=color, alpha=alpha)
    else:
        plt.plot(perturbation_x, perturbation_y, linewidth=linewidth, alpha=alpha)
        plt.plot(arrow[:, 0], arrow[:, 1], linewidth=linewidth, alpha=alpha)
    plt.axis('equal')
    return


def self_edge(p, linewidth=7.0, color=None, alpha=alpha_const):
    size = 0.4
    center = np.sqrt((1 + size)) * p
    radius = 0.8 * np.linalg.norm(center - p)
    t = np.linspace(0, 2 * np.pi - 0, 60)

    direction = center + radius * center / np.linalg.norm(center)
    radius_arrow = np.linalg.norm(direction)
    direction = direction / radius_arrow
    perpendicular = np.array([direction[1], -direction[0]])
    middle = center + radius * center / np.linalg.norm(center)
    arrow = 0.1 * np.array([perpendicular - direction, 0 * direction, perpendicular + direction]) + middle

    if color is None:
        plt.plot(center[0] + radius * np.sin(t), center[1] + radius * np.cos(t), linewidth=linewidth, alpha=alpha)
        plt.plot(arrow[:, 0], arrow[:, 1], linewidth=linewidth, alpha=alpha)
    else:
        plt.plot(center[0] + radius * np.sin(t), center[1] + radius * np.cos(t), linewidth=linewidth, color=color,
                 alpha=alpha)
        plt.plot(arrow[:, 0], arrow[:, 1], linewidth=linewidth, color=color, alpha=alpha)

    return


def define_nodes(n_nodes, rotation = 0.):
    y = np.cos(2 * np.pi / n_nodes * (rotation + np.linspace(0, n_nodes, n_nodes + 1))[1:])
    x = np.sin(2 * np.pi / n_nodes * (rotation + np.linspace(0, n_nodes, n_nodes + 1))[1:])
    return np.sqrt(n_nodes) * np.array([x, y])


def plot_nodes(xy, node_numbers=None):
    if node_numbers is None:
        node_numbers = range(np.shape(xy)[1])
    for i in range(np.shape(xy)[1]):
        plot_node(xy[:, i], number=str(node_numbers[i]))


def plot_node(node, number=None):
    plot_circle(center=node, radius=0.1, color='b', linewidth=21.0)
    if number:
        plt.text(node[0] - 0.04, node[1] - 0.04, number, color='w', size='x-large', fontweight='bold')


def plot_graph(adj, linewidth=7.0, color_repr='r', color_act='g'):
    n_nodes = np.shape(adj)[0]
    xy = define_nodes(n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj[i, j] < 0:
                plot_arch(xy[:, i], xy[:, j], color=color_repr, linewidth=linewidth*np.abs(adj[i, j]))
            elif adj[i, j] > 0:
                plot_arch(xy[:, i], xy[:, j], color=color_act, linewidth=linewidth*np.abs(adj[i, j]))
    plot_nodes(xy)
    plt.axis('equal')
    return

def define_tier_nodes(tiers, n_nodes):
    xy = np.empty((2, 0))
    radius = 1.
    for nodes in tiers:
        n_nodes_tier = len(nodes)
        xy = np.append(xy, radius * define_nodes(n_nodes_tier, rotation=(radius - 1) * 0.5 * np.pi), axis=1)
        radius += 1.
    return xy

def plot_tier_graph(tiers, adj, linewidth=7.0, color_repr='r', color_act='g'):
    n_nodes = np.shape(adj)[0]
    xy = define_tier_nodes(tiers, n_nodes)
    squeezed_tiers = [node for tier in tiers for node in tier]
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj[squeezed_tiers[i], squeezed_tiers[j]] < 0:
                plot_arch(xy[:, i], xy[:, j], color=color_repr, linewidth=linewidth*np.abs(adj[squeezed_tiers[i], squeezed_tiers[j]]))

            elif adj[squeezed_tiers[i], squeezed_tiers[j]] > 0:
                plot_arch(xy[:, i], xy[:, j], color=color_act, linewidth=linewidth*np.abs(adj[squeezed_tiers[i], squeezed_tiers[j]]))

    plot_nodes(xy, node_numbers=squeezed_tiers)
    plt.axis('equal')
    return


def simulated_annealing(length_opt, constraint, tiers= None, adj=None):
    base_config = np.array(range(length_opt))
    def new_config(old_config, temp):
        new_configuration = np.random.permutation(base_config)
        while np.linalg.norm(new_configuration - old_config) > temp:
            new_configuration = np.random.permutation(base_config)
        return new_configuration
    max_temp = length_opt**3
    configuration = np.random.permutation(base_config)
    constraint_val = constraint(configuration)
    for temp in range(max_temp, 0, -1):
        test_configuration = new_config(configuration, temp)
        if constraint(test_configuration) < constraint_val:
            configuration = test_configuration
            constraint_val = constraint(configuration)
            if __name__ == '__main__' and tiers is not None:
                opt_tiers = reshape_list(tiers, configuration)
                print(opt_tiers)
                plot_tier_graph(opt_tiers, adj)
                plt.show()

    return configuration

def reshape_list(shape_list, flat_list):
    # return a list with the shape of shape_list, but the elements of flat_list
    iterator = iter(flat_list)
    return [[next(iterator) for _ in sublist]
            for sublist in shape_list]

def optimize_tier_list(tiers, adj):
    n_nodes = np.shape(adj)[0]
    xy = define_tier_nodes(tiers, n_nodes) # the position of the nodes are fixed
    def sum_distance(squeezed_tiers):
        def position(node):
            index = next(idx for idx, elem in enumerate(squeezed_tiers) if elem == node)
            return xy[:,index]
        def distance(index1, index2):
            return np.linalg.norm(np.array([position(index1), position(index2)]))
        sum_dist = 0
        for i in range(n_nodes):
            sum_dist += np.sum([distance(i,j) for j in range(n_nodes) if adj[i,j]!=0])
        return - sum_dist
    opt_squeezed_tiers = simulated_annealing(n_nodes, sum_distance, tiers=tiers, adj=adj)
    opt_tiers = reshape_list(tiers, opt_squeezed_tiers)
    return opt_tiers



if __name__ == "__main__":

    plot_circle()
    plot_arch(np.array([1, 1]) / np.sqrt(2), np.array([1, 0.]))
    plot_arch(np.array([1, 0.]), np.array([1, 0.]))
    plt.show()

    n_nodes = 5
    xy = define_nodes(n_nodes)
    fig = plt.gcf()
    ax = fig.gca()
    plot_circle()
    for i in range(n_nodes - 1):
        plot_arch(xy[:, i], xy[:, i + 1])
    plot_arch(xy[:, -1], xy[:, 0])
    plot_nodes(xy)
    plt.axis('equal')
    plt.show()

    Adjacency = np.array([[1, 0, 1, 0], [0, -1, 0, 0], [1, -1, -1, 0], [0, -1, -1, 0]])
    plot_graph(Adjacency, linewidth=10.)
    # plt.show()

    Adjacency = np.array([[-0.5, 0, .1, 0], [0, -0.91, 0, 0], [1, -0.7, -0.6, 0], [0, -0.1, -0.41, 0]])
    plot_graph(Adjacency, linewidth=10.,  color_repr='yellow', color_act='cyan')
    plt.show()

    par_struct, adjacency = create_random_network(10)
    plot_graph(adjacency, linewidth=1.)
    plt.show()
    """
    # while the idea of a "tiered" graph is interesting, I don't seem to be able to find an algorithm that 
    # plots nice graphs automatically
    tier_list = [[1,2, 3], [4, 5]]
    adjacency = np.array([[1, 1, 1, 1, 0],
                          [0, 0, 1, 1, 1.],
                          [1, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0],
                          [1, 1, 0, 0, 0]
                          ])
    tier_list = optimize_tier_list(tier_list, adjacency)
    plot_tier_graph(tier_list, adjacency)
    plt.show()

    par_struct, adjacency = create_random_network(10)
    tier_list = [[0 ,2, 3], [5, 6, 7], [4, 8, 9, 1]]
    plot_tier_graph(tier_list, adjacency)
    plt.show()
    """
