import os
import pickle
import numpy as np
import pandas as pd


def read_metadata_file(file_path):
    data = pd.read_csv(file_path, delimiter=' ', header=None)
    return data.values


def create_metadata():
    data_folder = 'data/svm/'
    dest_folder = 'data/svm-h/'
    all_dataset_name = sorted(os.listdir(data_folder))

    for i, item in enumerate(all_dataset_name):
        data = read_metadata_file(data_folder+item)
        data = data[:, :7]
        # min_y, max_y = min(data[:, 0]), max(data[:, 0])
        # data[:, 0] = (data[:, 0] - min_y)/(max_y - min_y)
        data = pd.DataFrame(data)
        data.to_csv(dest_folder+item, header=False, index=False, sep=' ')


def process_resnet_metadata():
    datasets = list()
    datasets.extend([('result_' + item + '_100.pkl') for item in ['char74k', 'cifar-10', 'cifar-100', 'tiny-imagenet']])
    datasets.extend([('result_' + item + '_200.pkl') for item in ['plant_seedling', 'caltech101', 'caltech256',
                                                                  'dog_breed', 'dog_vs_cat', 'svhn']])
    dataset_names = ['char74k', 'cifar-10', 'cifar-100', 'tiny-imagenet']
    dataset_names.extend(['plant_seedling', 'caltech101', 'caltech256', 'dog_breed', 'dog_vs_cat', 'svhn'])
    for index, item in enumerate(datasets):
        dataset_name = dataset_names[index]
        meta_list = list()
        f = open('data/resnet/'+item, 'rb')
        data = pickle.load(f)
        for row in data:
            config, array, val_err, test_err = row
            row_list = list()
            row_list.append(1. - val_err)
            row_list.extend([1., 0.] if array[-3] == 0 else [0., 1.])
            row_list.extend(array[:-3])
            row_list.extend(array[-2:])
            meta_list.append(row_list)
        data = pd.DataFrame(np.array(meta_list))
        data.to_csv('data/resnet-h/' + dataset_name, header=False, index=False, sep=' ')


def process_xgb_metadata():
    datasets = sorted(os.listdir('data/xgb/'))
    for index, item in enumerate(datasets):
        dataset_name = item.split('.')[0]
        meta_list = list()
        f = open('data/xgb/'+item, 'rb')
        data = pickle.load(f)
        for row in data:
            config, array, val_acc = row
            row_list = list()
            row_list.append(val_acc)
            row_list.extend(array)
            meta_list.append(row_list)
        data = pd.DataFrame(np.array(meta_list))
        data.to_csv('data/xgb-h/' + dataset_name + '.txt', header=False, index=False, sep=' ')


if __name__ == "__main__":
    # create_metadata()
    # process_resnet_metadata()
    process_xgb_metadata()
