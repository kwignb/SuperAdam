import os, sys
from os.path import join, dirname

from omegaconf import OmegaConf

import numpy as np
import tensorflow_datasets as tfds

sys.path.append(join(dirname(__file__), ".."))


def create_dataset(cfg):
    
    n_classes = cfg.DATA.N_CLASSES
    target_classes = cfg.DATA.TARGET_CLASSES
    save_dir = cfg.DATA.SAVE_DIR
    dataset_name = cfg.DATA.DATASET_NAME
    n_train = str(cfg.DATA.TRAIN_SIZE)
    n_test = str(cfg.DATA.TEST_SIZE)
    
    x_train, y_train, x_test, y_test = _load_tf_dataset(cfg)
    
    n_classes = y_train.shape[-1]
    if n_classes == 2:
        y_train = np.array([[0.5 * (c1 - c2)] for c1, c2 in y_train])
        y_test = np.array([[0.5 * (c1 - c2)] for c1, c2 in y_test])
        
    if target_classes is None:
        target_classes = list(range(n_classes))
        
    for i in range(len(x_train)):
        x_train[i] /= np.linalg.norm(x_train[i])
    for i in range(len(x_test)):
        x_test[i] /= np.linalg.norm(x_test[i])
        
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f'{dataset_name}_train{n_train}_test{n_test}_{str(n_classes)}classes'
        path = os.path.join(save_dir, file_name)
        np.savez(path, x_train, y_train, x_test, y_test, target_classes)
        
    return x_train, y_train, x_test, y_test, target_classes
        
        
def _load_tf_dataset(cfg):
    
    dataset_name = cfg.DATA.DATASET_NAME
    target_classes = cfg.DATA.TARGET_CLASSES
    n_train, n_test = cfg.DATA.TRAIN_SIZE, cfg.DATA.TEST_SIZE
    flatten, normalize = cfg.DATA.FLATTEN, cfg.DATA.NORMALIZE
    
    ds_builder = tfds.builder(dataset_name)
    
    ds_train, ds_test = tfds.as_numpy(tfds.load(
        dataset_name, split=['train', 'test'], data_dir='tensorflow_datasets',
        batch_size=-1, as_dataset_kwargs={'shuffle_files': False}
    ))
    
    train_images, train_labels, test_images, test_labels = \
        (ds_train['image'], ds_train['label'], ds_test['image'], ds_test['label'])
    
    num_classes = ds_builder.info.features['label'].num_classes
    
    if target_classes is not None:
        for target in target_classes:
            assert 0 <= target < num_classes
        num_classes = len(target_classes)
        
        def extract_subset(images, labels):
            new_images, new_labels = [], []
            for image, label in zip(images, labels):
                if label in target_classes:
                    idx = target_classes.index(label)
                    new_images.append(image)
                    new_labels.append(idx)
            return np.array(new_images), np.array(new_labels)
    
        train_images, train_labels = extract_subset(train_images, train_labels)
        test_images, test_labels = extract_subset(test_images, test_labels)
    
    train_images, train_labels = train_images[:n_train], train_labels[:n_train]
    test_images, test_labels = test_images[:n_test], test_labels[:n_test]
    
    if flatten:
        train_images = _partial_flatten(train_images)
        test_images = _partial_flatten(test_images)
    
    if normalize:
        train_images = _partial_normalize(train_images)
        test_images = _partial_normalize(test_images)
    
    train_labels = _one_hot(train_labels, num_classes)
    test_labels = _one_hot(test_labels, num_classes)
    
    return train_images, train_labels, test_images, test_labels
    
    
def _partial_flatten(x):
    return np.reshape(x, (x.shape[0], -1))


def _partial_normalize(x):
    return (x - np.mean(x)) / np.std(x)


def _one_hot(x, k, dtype=np.float32):
    return np.array(x[:, None] == np.arange(k), dtype)


def read_yaml(yaml_path):
    return OmegaConf.load(yaml_path)
