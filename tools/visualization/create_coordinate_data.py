import os
import csv
import pickle
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter
dataset_names = ['caltech101', 'char74k', 'cifar-10', 'cifar-100', 'dog_breed', 'plant_seedling',
            'caltech256', 'dog_vs_cat', 'tiny-imagenet']

datasets = [('result_' + item + '_30.pkl') for item in dataset_names]
file_path = 'data/visualization.csv'


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


def get_visual_csv(transformed=False, val=True):
    hp_names = ['val_acc' if val else 'test_acc']
    cs = create_configspace()
    hp_names.extend(cs.get_hyperparameter_names())
    hp_names.append('dataset')

    if not os.path.exists(file_path):
        indexs = list()
        for hp in hp_names[1:-1]:
            indexs.append(cs._hyperparameter_idx[hp])

        metadata = []
        for dataset_id, item in enumerate(datasets):
            f = open('data/resnet/'+item, 'rb')
            data = pickle.load(f)
            for row in data:
                config, array, val_err, test_err = row
                row_list = list()
                row_list.append((1. - val_err) if val else (1. - test_err))

                if not transformed:
                    for hp in hp_names[1:-1]:
                        row_list.append(config[hp] if not isinstance(config[hp], bool) else int(config[hp]))
                else:
                    for index in indexs:
                        row_list.append(array[index])
                row_list.append(dataset_id)
                metadata.append(row_list)

        with open(file_path, 'w+') as outcsv:
            # configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(hp_names)
            for item in metadata:
                # Write item to outcsv
                writer.writerow(item)
    return file_path, hp_names, dataset_names


if __name__ == "__main__":
    print(get_visual_csv())
