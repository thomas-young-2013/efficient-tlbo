import pickle
import numpy as np

datasets = list()
datasets.extend([('result_' + item + '_100.pkl') for item in ['char74k', 'cifar-10', 'cifar-100', 'tiny-imagenet']])
datasets.extend([('result_' + item + '_200.pkl') for item in ['plant_seedling', 'caltech101', 'caltech256',
                                                              'dog_breed', 'dog_vs_cat', 'svhn']])


def load_resnet_metadata(test_id, num_train_instances=50):
    num_test = len(datasets)
    test_id = test_id % num_test
    meta_datasets = []
    test_meta_dataset = None
    for index, item in enumerate(datasets):
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
        if index == test_id:
            test_meta_dataset = np.array(meta_list)
        else:
            meta_data = np.array(meta_list)[:num_train_instances]
            meta_datasets.append(meta_data)
    assert test_meta_dataset is not None
    return meta_datasets, test_meta_dataset
