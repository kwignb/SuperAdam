import os, sys
from os.path import join, dirname

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

sys.path.append(join(dirname(__file__), ".."))


def vis_one_df(df_path, optim_name, one_epoch=True):
    
    df = pd.read_csv(df_path, index_col=0)
    
    if one_epoch:
        df = df.iloc[1:]
    
    loss_list = ['epoch', 'train_loss', 'test_loss']
    acc_list = ['epoch', 'train_accuracy', 'test_accuracy']
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    
    df[loss_list].plot(x='epoch', style=['r', 'b'], ax=axes[0])
    df[acc_list].plot(x='epoch', style=['r', 'b'], ax=axes[1])
    
    axes[0].set_ylabel('loss')
    axes[1].set_ylabel('accuracy')
    
    fig.suptitle(f'Optimizer: {optim_name}')
    fig.tight_layout()
    fig.show()
    
    
def vis_two_df(df_path1, df_path2, opt_name1, opt_name2, one_epoch=True):
    
    df1 = pd.read_csv(df_path1, index_col=0)
    df2 = pd.read_csv(df_path2, index_col=0)
    
    if one_epoch:
        df1, df2 = df1.iloc[1:], df2.iloc[1:]
    
    df1_train_loss, df1_test_loss = df1['train_loss'], df1['test_loss']
    df1_train_acc, df1_test_acc = df1['train_accuracy'], df1['test_accuracy']
    
    df2_train_loss, df2_test_loss = df2['train_loss'], df2['test_loss']
    df2_train_acc, df2_test_acc = df2['train_accuracy'], df2['test_accuracy']
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
    
    axes[0].plot(df1_train_loss, label=f'train loss ({opt_name1})', color='r')
    axes[0].plot(df1_test_loss, label=f'test loss ({opt_name1})', color='b')
    ax2 = axes[0].twinx()
    ax2.plot(df2_train_loss, label=f'train loss ({opt_name2})', color='r', linestyle='dashed')
    ax2.plot(df2_test_loss, label=f'test loss ({opt_name2})', color='b', linestyle='dashed')
    ax2.set_ylabel(f'{opt_name2}: loss', fontsize=18)
    
    axes[1].plot(df1_train_acc, label=f'train accuracy ({opt_name1})', color='r')
    axes[1].plot(df1_test_acc, label=f'test accuracy ({opt_name1})', color='b')
    axes[1].plot(df2_train_acc, label=f'train accuracy ({opt_name2})', color='r', linestyle='dashed')
    axes[1].plot(df2_test_acc, label=f'test accuracy ({opt_name2})', color='b', linestyle='dashed')
    
    axes[0].set_xlabel('epoch', fontsize=18)
    axes[1].set_xlabel('epoch', fontsize=18)
    axes[0].set_ylabel(f'{opt_name1}: loss', fontsize=18)
    axes[1].set_ylabel('accuracy', fontsize=18)
    
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[0].legend(h1+h2, l1+l2, loc='lower center', bbox_to_anchor=(.5, 1.01), ncols=2, fontsize=12)
    axes[1].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncols=2, fontsize=12)
    
    fig.tight_layout()
    fig.show()
    
    
def vis_multi_df(df_paths, opt_names, one_epoch=True):
    
    dfs = [pd.read_csv(df_path, index_col=0) for df_path in df_paths]
    
    if one_epoch:
        dfs = [df.iloc[1:] for df in dfs]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
    
    colors = ['r', 'b', 'g', 'm', 'k']
    
    for i, (df, opt_name, color) in enumerate(zip(dfs, opt_names, colors[:len(dfs)])):
        axes[0,0].plot(
            df['train_loss'], label=f'train loss ({opt_name})', color=color
            )
        axes[0,1].plot(
            df['test_loss'], label=f'test loss ({opt_name})', color=color
        )
        axes[1,0].plot(
            df['train_accuracy'], label=f'train accuracy ({opt_name})', color=color
            )
        axes[1,1].plot(
            df['test_accuracy'], label=f'test accuracy ({opt_name})', color=color
        )
    
    axes[0,0].set_xlabel('Epochs', fontsize=18)
    axes[0,1].set_xlabel('Epochs', fontsize=18)
    axes[1,0].set_xlabel('Epochs', fontsize=18)
    axes[1,1].set_xlabel('Epochs', fontsize=18)
    
    axes[0,0].set_ylabel('Train Loss', fontsize=18)
    axes[0,1].set_ylabel('Test Loss', fontsize=18)
    axes[1,0].set_ylabel('Train Accuracy', fontsize=18)
    axes[1,1].set_ylabel('Test Accuracy', fontsize=18)
    
    axes[0,0].legend(fontsize=15)
    axes[0,1].legend(fontsize=15)
    axes[1,0].legend(fontsize=15)
    axes[1,1].legend(fontsize=15)
    
    fig.tight_layout()
    fig.show()