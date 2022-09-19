from random import shuffle
from ..L0_Linear import L0UnstructuredLinear
import torch
import torch.nn as nn
import numpy as np
from .utils import train_L0_Linear, L0_Loss, train_L0_probe

def test_training_vs_test_mask():
    l0_layer = L0UnstructuredLinear(in_features=5, out_features=10)
    l0_layer.train(True)

    train_mask = l0_layer.mask()
    train_mask = train_mask.reshape(-1)
    assert((torch.sum(train_mask == 1.0) + torch.sum(train_mask == 0.0)) != len(train_mask))

    l0_layer.train(False)

    train_mask = l0_layer.mask()
    train_mask = train_mask.reshape(-1)
    assert((torch.sum(train_mask == 1.0) + torch.sum(train_mask == 0.0)) == len(train_mask))

def test_stretching():
    low_stretch = L0UnstructuredLinear(in_features=5, out_features=10, stretch=.01)
    mask = low_stretch.mask()
    low_stretch_exact_extremes = torch.sum(mask == 1.0) + torch.sum(mask == 0.0)

    high_stretch = L0UnstructuredLinear(in_features=5, out_features=10, stretch=.5)
    mask = high_stretch.mask()
    high_stretch_exact_extremes = torch.sum(mask == 1.0) + torch.sum(mask == 0.0)
    assert(high_stretch_exact_extremes > low_stretch_exact_extremes)

def test_init_mean_L0_norm_relationship():
    # Mean is equivalent to parameter drop rate, low mean should mean > L0 norm
    low_mean = L0UnstructuredLinear(in_features=5, out_features=10, init_mean=.1)
    low_mean_norm = low_mean.mask.l0_norm()
    high_mean = L0UnstructuredLinear(in_features=5, out_features=10, init_mean=.9)
    high_mean_norm = high_mean.mask.l0_norm()
    assert(low_mean_norm > high_mean_norm)

def test_sampling_convergence():
    # When initing with a 50% dropout rate, the expected norm should be 25
    expected_norm = 25
    N_samples = 50
    true_norms = []
    for _ in range(N_samples):
        l0_layer = L0UnstructuredLinear(in_features=5, out_features=10, init_mean=.5)
        l0_layer.train(False)
        mask = l0_layer.mask()
        mask = mask.reshape(-1)
        emp_norm = torch.sum(mask == 1.0)
        true_norms.append(emp_norm)
        l0_layer.compiled_weight = None
    empirical_norm = np.mean(true_norms)
    assert(expected_norm - 3 < empirical_norm < expected_norm + 3)

    # When calculating L0 loss, there is a bias term (eq 12 in Louizos et al.) in addition to logalpha,
    # making the l0 loss > than the empirical norm after discretizing the gate
    loss_norm = l0_layer.mask.l0_norm()
    assert(loss_norm > empirical_norm)

def test_l0_training():
    model, acc = train_L0_Linear()
    assert(acc > .95)
    model.train(False)
    active_params = L0_Loss()._get_model_l0(model)
    max_active_params = L0_Loss()._get_model_prunable_params(model)
    assert(active_params < max_active_params)

def test_diff_lambda_training():
    model, _ = train_L0_Linear(lambda_numerator=.1)
    model.train(False)
    low_lambda_active_params = L0_Loss()._get_model_l0(model)

    model, _ = train_L0_Linear(lambda_numerator=5)
    model.train(False)
    high_lambda_active_params = L0_Loss()._get_model_l0(model)
    assert(high_lambda_active_params < low_lambda_active_params)

def test_pretrain_prune():
    mlp_model, unpruned_acc, l0_mlp, pruned_acc = train_L0_probe()
    l0_mlp.train(False)
    max_params = L0_Loss()._get_model_prunable_params(l0_mlp)
    active_params = L0_Loss()._get_model_l0(l0_mlp)
    print(f'Params: {active_params} / {max_params}')
    assert(max_params > active_params)
    assert(pruned_acc > unpruned_acc - .05)
    assert(torch.equal(mlp_model.layer_0.weight,l0_mlp.layer_0.weight.T))
