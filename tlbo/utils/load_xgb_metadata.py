import os
import pickle
import numpy as np


def load_xgb_metadata(test_id, ratio=1.):
    num_train_instances = int(50*ratio)
    datasets = sorted(os.listdir('data/xgb/'))
    test_id = test_id % len(datasets)
    meta_datasets = []
    test_meta_dataset = None
    for index, item in enumerate(datasets):
        meta_list = list()
        f = open('data/xgb/'+item, 'rb')
        data = pickle.load(f)
        for row in data:
            config, array, val_acc = row
            row_list = list()
            row_list.append(val_acc)
            row_list.extend(array)
            meta_list.append(row_list)
        if index == test_id:
            test_meta_dataset = np.array(meta_list)
        else:
            meta_data = np.array(meta_list)[:num_train_instances]
            meta_datasets.append(meta_data)
    assert test_meta_dataset is not None
    return meta_datasets, test_meta_dataset
