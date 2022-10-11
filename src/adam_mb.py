import os, sys, time
from os.path import join, dirname

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.example_libraries import optimizers

from neural_tangents import stax

sys.path.append(join(dirname(__file__), ".."))
from src.utils import read_yaml, create_dataset, load_dataset

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def accuracy(y, y_hat):
    if y_hat.shape[-1] == 1:
        return jnp.mean(jnp.sign(y) == jnp.sign(y_hat))
    return jnp.mean(jnp.argmax(y, axis=1) == jnp.argmax(y_hat, axis=1))


def train_batch(grad_fn, params, batch_x, batch_y, G):
    grads = grad_fn(params, batch_x, batch_y, G)
    


def main():
    
    global opt_state
    
    # load config
    cfg = read_yaml(yaml_path='src/configs/adam.yaml')
    
    # parameters in config
    dataset_name = cfg.DATA.DATASET_NAME
    n_classes = cfg.DATA.N_CLASSES
    target_classes = cfg.DATA.TARGET_CLASSES
    train_num = cfg.DATA.TRAIN_SIZE
    test_num = cfg.DATA.TEST_SIZE
    use_npz = cfg.DATA.USE_NPZ
    
    n_layers = cfg.MODEL.N_LAYERS
    n_width = cfg.MODEL.N_WIDTH
    weight_variance = cfg.MODEL.WEIGHT_VARIANCE
    bias_variance = cfg.MODEL.BIAS_VARIANCE
    
    batch_size = cfg.OPTIMIZER.BATCH_SIZE
    learning_rate = cfg.OPTIMIZER.LEARNING_RATE
    
    epochs = cfg.GENERAL.EPOCHS
    devices = cfg.GENERAL.DEVICES
    random_seed = cfg.GENERAL.SEED
    
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
    
    if use_npz:
        x_train, y_train, x_test, y_test, target_classes = load_dataset(cfg)
    else:
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
    
    init_fn, apply_fn, _ = stax.serial(*_layers)
    
    key = random.PRNGKey(random_seed)
    _, params = init_fn(key, (-1, x_train.shape[-1]))
    
    opt_init, opt_apply, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(params)
        
    @jax.jit
    def _loss(params, x, y):
        fx = apply_fn(params, x)
        loss = 0.5 * jnp.mean(jnp.sum((fx - y)**2, axis=1))
        return loss
    
    def train_batch(epoch, batch_idx, params, batch_x, batch_y):
        global opt_state
        if epoch == 0:
            return params
        else:
            grads = grad_fn(params, batch_x, batch_y)
            opt_state = opt_apply(batch_idx, grads, opt_state)
            params = get_params(opt_state)
            return params
    
    def train_one_epoch(epoch, params, x_train, y_train):
        idx = random.permutation(key, train_num)
        num_batches = train_num // batch_size
        batch_losses, batch_accs = [], []
        for batch in range(num_batches):
            batch_idx = idx[batch * batch_size: (batch + 1) * batch_size]
            batch_x, batch_y = x_train[batch_idx], y_train[batch_idx]
            params = train_batch(epoch, batch, params, batch_x, batch_y)
            batch_loss = _loss(params, batch_x, batch_y)
            batch_acc = accuracy(apply_fn(params, batch_x), batch_y)
            batch_losses.append(batch_loss)
            batch_accs.append(batch_acc)
        return params, np.mean(batch_losses), np.mean(batch_accs)

    def valid_batch(params, batch_x, batch_y):
        batch_loss = _loss(params, batch_x, batch_y)
        batch_acc = accuracy(apply_fn(params, batch_x), batch_y)
        return batch_loss, batch_acc

    def valid_one_epoch(params, x_test, y_test):
        idx = random.permutation(key, test_num)
        num_batches = test_num // batch_size
        batch_losses, batch_accs = [], []
        for batch in range(num_batches):
            batch_idx = idx[batch * batch_size: (batch + 1) * batch_size]
            batch_x, batch_y = x_test[batch_idx], y_test[batch_idx]
            batch_loss, batch_acc = valid_batch(params, batch_x, batch_y)
            batch_losses.append(batch_loss)
            batch_accs.append(batch_acc)
        return np.mean(batch_losses), np.mean(batch_accs)
        
    # Define the gradient
    grad_fn = jit(lambda params, x, y: grad(_loss)(params, x, y))
    
    print('========================')
    for _, vals in cfg.items():
        for k, v in vals.items():
            print(f'{k}: {v}')
    print('========================')
    
    print(f'Training for {epochs} epochs.')
    
    entries = [
        'epoch', 'train_accuracy','train_loss',
        'test_accuracy', 'test_loss', 'time'
        ]
    entry_widths = [max(11, len(s)) for s in entries]
    templates = []
    for entry, w in zip(entries, entry_widths):
        templates.append((entry, '{:<%dg}  ' % w, ' ' * (w+2)))

    header = '  '.join(('{:%d}' % w for w in entry_widths)).format(*entries)
    print(header)
    
    results = []
    for epoch in range(epochs+1):
        
        start_time = time.time()
        
        params, train_loss, train_acc = train_one_epoch(
            epoch, params, x_train, y_train
            )
        test_loss, test_acc = valid_one_epoch(
            params, x_test, y_test
        )
        
        epoch_time = time.time() - start_time
        
        log = {
            'epoch': epoch,
            'train_accuracy': train_acc,
            'train_loss': train_loss,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'time': epoch_time
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
    df_name = f'adam_{dataset_name}_{batch_size}'
    df.to_csv(f'results/{df_name}.csv')
    
if __name__ == '__main__':
    main()