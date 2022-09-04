import os, sys
from os.path import join, dirname

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random
from jax.example_libraries import optimizers

from neural_tangents import stax

sys.path.append(join(dirname(__file__), ".."))
from src.utils import read_yaml, create_dataset

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def accuracy(y, y_hat):
    if y_hat.shape[-1] == 1:
        return jnp.mean(jnp.sign(y) == jnp.sign(y_hat))
    return jnp.mean(jnp.argmax(y, axis=1) == jnp.argmax(y_hat, axis=1))


def main():
    
    # load config
    cfg = read_yaml(yaml_path='src/configs/generalized_adam.yaml')
    
    # parameters in config
    n_classes = cfg.DATA.N_CLASSES
    target_classes = cfg.DATA.TARGET_CLASSES
    
    n_layers = cfg.MODEL.N_LAYERS
    n_width = cfg.MODEL.N_WIDTH
    weight_variance = cfg.MODEL.WEIGHT_VARIANCE
    bias_variance = cfg.MODEL.BIAS_VARIANCE
    
    batch_size = cfg.OPTIMIZER.BATCH_SIZE
    learning_rate = cfg.OPTIMIZER.LEARNING_RATE
    
    epochs = cfg.GENERAL.EPOCHS
    devices = cfg.GENERAL.DEVICES
    random_seed = cfg.GENERAL.SEED
    store_on_device = cfg.GENERAL.STORE_ON_DEVICE
    
    # setup device
    if devices is None:
        devices = jax.device_count()
    
    # build data pipelines
    print('Loading data...')
    assert n_classes >= 2
    
    if target_classes is None:
        target_classes = list(range(n_classes))
    else:
        target_classes = [int(cls) for cls in target_classes]
    
    assert len(target_classes) == n_classes
    
    x_train, y_train, x_test, y_test, target_classes = create_dataset(cfg)
    
    if n_classes == 2:
        n_outputs = 1
    else:
        n_outputs = n_classes
    
    # build the network (TODO: Adapt CNN)
    _layers = []
    assert n_layers > 1
    w_std = jnp.sqrt(weight_variance)
    b_std = jnp.sqrt(bias_variance)
    for i in range(n_layers - 1):
        _layers += [
            stax.Dense(n_width, W_std=w_std, b_std=b_std, parameterization='ntk'),
            stax.Relu()
        ]
    _layers.append(
        stax.Dense(n_outputs, W_std=w_std, b_std=b_std, parameterization='ntk')
        )
    
    init_fn, apply_fn, kernel_fn = stax.serial(*_layers)
    
    key = random.PRNGKey(random_seed)
    _, params = init_fn(key, (-1, x_train.shape[-1]))
    
    opt_init, opt_apply, get_params = optimizers.sgd(learning_rate)
    opt_state = opt_init(params)
    
    loss = lambda params, x, y, ntk: 0.5 * jnp.mean(
        (apply_fn(params, x) - y).T @ jnp.linalg.inv(ntk) @ (apply_fn(params, x) - y)
        )
        
    print('========================')
    for keys, vals in cfg.items():
        for key, val in vals.items():
            print(f'{key}: {val}')
    print('========================')
    
    # Get initial values of the network in function space.
    fx0_train = apply_fn(params, x_train)
    fx0_test = apply_fn(params, x_test)
    
    # Get Neural Tangent Kernels
    ntk_train = kernel_fn(x_train, x_train, 'ntk')
    ntk_test = kernel_fn(x_test, x_test, 'ntk')
    
    # Define the gradient
    grad_fn = lambda params, x, y, ntk: jax.grad(loss)(params, x, y, ntk)
    
    print(f'Training for {epochs} epochs.')
    
    entries = ['epoch', 'train_accuracy', 'train_loss', 'test_accuracy', 'test_loss']
    entry_widths = [max(11, len(s)) for s in entries]
    templates = []
    for entry, w in zip(entries, entry_widths):
        templates.append((entry, '{:<%dg}  ' % w, ' ' * (w+2)))

    header = '  '.join(('{:%d}' % w for w in entry_widths)).format(*entries)
    print(header)
    
    results = []
    for i in range(epochs):
        if i == 0:
            # init values
            fx_train, fx_test = fx0_train, fx0_test
        else:
            args = [params, x_train, y_train, ntk_train]
            grads = grad_fn(*args)
            opt_state = opt_apply(i, grads, opt_state)
            params = get_params(opt_state)
            fx_train = apply_fn(params, x_train)
            fx_test = apply_fn(params, x_test)

        train_loss = loss(params, x_train, y_train, ntk_train).tolist()
        train_acc = accuracy(fx_train, y_train).tolist()
        test_loss = loss(params, x_test, y_test, ntk_test).tolist()
        test_acc = accuracy(fx_test, y_test).tolist()
        
        log = {
            'epoch': i+1,
            'train_accuracy': train_acc,
            'train_loss': train_loss,
            'test_accuracy': test_acc,
            'test_loss': test_loss
        }
        
        # print report
        report, last_entry = '', ''
        for entry, template, empty in templates:
            if entry in log:
                report += template.format(log[entry])
            else:
                report += empty
            last_entry = entry
        print(report)
                
        results.append(log)
        
    if not os.path.exists('results'):
        os.makedirs('results')
        
    df = pd.json_normalize(results)
    df.to_csv('results/generalized_adam.csv')
    
if __name__ == '__main__':
    main()