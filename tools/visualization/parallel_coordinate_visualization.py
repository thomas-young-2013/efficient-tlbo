import sys
import plotly
import argparse
import plotly.graph_objs as go
import pandas as pd

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar-100')
parser.add_argument('--metric', choices=['val_acc', 'test_acc'], default='val_acc')
parser.add_argument('--mode', choices=['local', 'server'], default='local')
args = parser.parse_args()
if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/RTL')
from tools.visualization.create_coordinate_data import get_visual_csv

# batch_size,learning_rate,lr_decay_factor,momentum,nesterov,padding_size,weight_decay,dataset
# ['caltech101', 'char74k', 'cifar-10', 'cifar-100', 'dog_breed', 'plant_seedling',
# 'caltech256', 'dog_vs_cat', 'tiny-imagenet']


def example_one(file_path, hp_names, dataset_names, dataset_name='cifar-100', metric='val_acc'):
    index = dataset_names.index(dataset_name)
    df = pd.read_csv(file_path)
    df = df.loc[df['dataset'] == index]

    data = [
        go.Parcoords(
            line=dict(color=df[metric],
                      colorscale='Jet',
                      showscale=True,
                      reversescale=True,
                      cmin=df[metric].min(),
                      cmax=df[metric].max()),
            dimensions=list([
                dict(range=[32, 256],
                     label='Batch Size', values=df['batch_size']),
                dict(range=[0.01, 0.5],
                     label='Learning Rate', values=df['learning_rate']),
                dict(range=[0.05, 0.5],
                     label='LR Reduction', values=df['lr_decay_factor']),
                dict(range=[1e-6, .95],
                     label='Momentum', values=df['momentum']),
                dict(tickvals=[0, 1],
                     ticktext=['True', 'False'],
                     label='Nesterov', values=df['nesterov']),
                dict(range=[0, 6],
                     ticktext=[1, 2, 3, 4, 5],
                     label='Padding Size', values=df['padding_size']),
                dict(range=[1e-5, 1e-3],
                     label='Weight Decay', values=df['weight_decay']),
                dict(range=[df[metric].min(), df[metric].max()],
                     label='Val Acc', values=df['val_acc']),
            ])
        )
    ]

    layout = go.Layout(
        plot_bgcolor='#E5E5E5',
        paper_bgcolor='#E5E5E5'
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='data/resnet-%s' % dataset_name)

    # plotly.offline.plot(data, filename='resnet-%s' % dataset_name)


def example_all(file_path, hp_names, dataset_names):
    df = pd.read_csv(file_path)
    data = [
        go.Parcoords(
            line=dict(color=df['val_acc'],
                      colorscale='Jet',
                      showscale=True,
                      reversescale=True,
                      cmin=0,
                      cmax=1.),
            dimensions=list([
                dict(tickvals=list(range(len(dataset_names))),
                     ticktext=dataset_names,
                     label='Dataset', values=df['dataset']),
                dict(range=[32, 256],
                     label='Batch Size', values=df['batch_size']),
                dict(range=[0.01, 0.5],
                     label='Learning Rate', values=df['learning_rate']),
                dict(range=[0.05, 0.5],
                     label='LR Reduction', values=df['lr_decay_factor']),
                dict(range=[1e-6, .95],
                     label='Momentum', values=df['momentum']),
                dict(tickvals=[0, 1],
                     ticktext=['True', 'False'],
                     label='Nesterov', values=df['nesterov']),
                dict(range=[0, 6],
                     ticktext=[1, 2, 3, 4, 5],
                     label='Padding Size', values=df['padding_size']),
                dict(range=[1e-5, 1e-3],
                     label='Weight Decay', values=df['weight_decay']),
                dict(range=[0.0, 1.],
                     label='Val Acc', values=df['val_acc']),
            ])
        )
    ]

    layout = go.Layout(
        plot_bgcolor='#E5E5E5',
        paper_bgcolor='#E5E5E5'
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='data/resnet-all.html')
    plotly.io.write_image(fig, 'data/resent-all.pdf', width=800, height=500)


def create_configspace():
    cs = ConfigurationSpace()
    lr = UniformFloatHyperparameter("learning_rate", 1e-2, 0.5, default_value=0.1, q=1e-2, log=True)
    momentum = UniformFloatHyperparameter("momentum", 1e-6, .95, default_value=.9)
    weight_decay = UniformFloatHyperparameter("weight_decay", 1e-5, 1e-3, default_value=1e-4, q=1e-5)
    nesterov = CategoricalHyperparameter("nesterov", [True, False], default_value=False)
    batch_size = UniformFloatHyperparameter("batch_size", 32, 256, q=16., default_value=128)
    padding_size = UniformFloatHyperparameter("padding_size", 1, 5, q=1., default_value=2)
    lr_reductions = UniformFloatHyperparameter("lr_decay_factor", 0.05, 0.5, default_value=0.1, q=0.05)
    cs.add_hyperparameters([lr, nesterov, momentum, lr_reductions, batch_size, weight_decay, padding_size])
    return cs


def test_hp():
    cs = create_configspace()
    default = cs.get_default_configuration()
    print(default.keys())
    print(cs.get_hyperparameter_names())
    print(cs.get_hyperparameter_names() == default.keys())
    print(default.get_array())
    print(default.get_dictionary())
    print(cs._hyperparameter_idx)
    print(default.get_array()[cs._hyperparameter_idx['weight_decay']])
    print(default.get('weight_decay'))
    for item in cs.sample_configuration(5):
        print(item.get_dictionary())
        print(item.get_array())


if __name__ == "__main__":
    data = get_visual_csv()
    # example_all(data[0], data[1], data[2])
    example_one(data[0], data[1], data[2], dataset_name=args.dataset, metric=args.metric)
