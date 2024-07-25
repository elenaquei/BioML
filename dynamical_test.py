import torch
import matplotlib.pyplot as plt
import numpy as np
from models.training import create_dataloader, maskedTrainer, doublebackTrainer
import scipy.io as io
from models.neural_odes import NeuralODE


# TODO: confirm this is the correct use of time_steps
def one_layer(nODE: NeuralODE):
    return nODE.time_steps == 1


def weight_norm_distance(nODE1: NeuralODE, nODE2: NeuralODE):
    if not one_layer(nODE1) or not one_layer(nODE2):
        raise Exception("Sorry, only single layer nODEs are treated")
    if not nODE1.architecture == nODE2.architecture:
        raise Exception("The two nODEs must have the same architecture")
    if nODE2.architecture is not 1:
        weight1, weight2 = extract_linear_layer(nODE1).weight, extract_linear_layer(nODE2).weight
        return np.linalg.norm(weight1 - weight2)

    weight1, weight2 = extract_linear_layer(nODE1).weight, extract_linear_layer(nODE2).weight
    return np.linalg.norm(weight1[0] - weight2[0]) + np.linalg.norm(weight1[1] - weight2[1])


def n_stable_eqs(nODE: NeuralODE):
    if not one_layer(nODE):
        raise Exception("Sorry, only single layer nODEs are treated")
    return nODE.time_steps == 1


def n_stable_periodic_orbits(nODE: NeuralODE):
    if not one_layer(nODE):
        raise Exception("Sorry, only single layer nODEs are treated")
    return False


def extract_linear_layer(nODE: NeuralODE):
    if not one_layer(nODE):
        raise Exception("Sorry, only single layer nODEs are treated")

    if nODE.architecture == 2 or nODE.architecture == 3:
        return nODE.fc2_time

    if nODE.architecture == 4:  # grn architecture : W sigma ( x + b )
        return nODE.fc2_time

    if nODE.architecture == 1:
        # -- R^{d_aug} -> R^{d_hid} layer --
        return [nODE.fc1_time, nODE.fc3_time]
    else:
        # -- R^{d_hid} -> R^{d_hid} layer --
        return nODE.fc2_time
