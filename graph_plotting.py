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


def define_nodes(n_nodes):
    y = np.cos(2 * np.pi / n_nodes * np.linspace(0, n_nodes, n_nodes + 1)[1:])
    x = np.sin(2 * np.pi / n_nodes * np.linspace(0, n_nodes, n_nodes + 1)[1:])
    return np.sqrt(n_nodes) * np.array([x, y])


def plot_nodes(xy):
    for i in range(np.shape(xy)[1]):
        plot_node(xy[:, i], number=str(i))


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
