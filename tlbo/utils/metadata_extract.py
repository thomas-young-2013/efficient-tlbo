import os
import numpy as np
import pandas as pd

'''
weka:
    hp_range = 103
    hp_indicator = 64
svm:
    hp_range = 6
    hp_indicator = 3
'''


def sample_metadata(metadata_list, trial_id, proportion=0.2):
    metadata_container = list()
    # keep the training data the same across methods.
    np.random.seed(trial_id)
    for metadata in metadata_list:
        meta_size = metadata.shape[0]
        indexes = list(range(meta_size))
        np.random.shuffle(indexes)
        indexes = indexes[: int(meta_size*proportion)]
        metadata_container.append(metadata[indexes])
    return metadata_container


def read_metadata_file(file_path):
    data = pd.read_csv(file_path, delimiter=' ', header=None)
    return data.values


def sparse_extraction(train_metadata, hp_range, hp_indicator, dense=True, metafeature=False):
    hp_index = list()
    if not dense:
        # Build reverse index.
        for i in range(hp_range - hp_indicator):
            hp_mapping = dict()
            hp_values = list()
            for row in train_metadata[0]:
                if hp_values.count(row[hp_indicator+1+i]) == 0:
                    hp_values.append(row[hp_indicator+1+i])
            hp_values = sorted(hp_values)
            for index, item in enumerate(hp_values):
                hp_mapping[item] = index
            hp_index.append(hp_mapping)

    # Remove rows in each metadata set.
    sparse_train_metadata = list()
    dense_train_metadata = list()
    for metadata in train_metadata:
        row_ids = set(list(range(0, len(metadata))))
        deleted_row_ids = []
        if not dense:
            for row_id, row in enumerate(metadata):
                for i in range(len(hp_index)):
                    value = row[hp_indicator+1+i]
                    if (hp_index[i][value] + 2) % 3 != 0:
                        if hp_indicator == 0 or (hp_indicator != 0 and value != 0):
                            deleted_row_ids.append(row_id)
                            break
        # Convert to a minimize problem.
        if metafeature:
            tmp_metadata = metadata[list(row_ids - set(deleted_row_ids)), :]
        else:
            tmp_metadata = metadata[list(row_ids - set(deleted_row_ids)), :hp_range + 1]

        # Normalize the output in train metadata.
        # acc_max = np.max(tmp_metadata[:, 0])
        # acc_min = np.min(tmp_metadata[:, 0])
        # tmp_metadata[:, 0] = (tmp_metadata[:, 0] - acc_min)/(acc_max - acc_min)
        sparse_train_metadata.append(tmp_metadata)
        if metafeature:
            dense_train_metadata.append(metadata)
        else:
            dense_train_metadata.append(metadata[:, :hp_range+1])
    if not dense:
        print('sparse size', len(sparse_train_metadata[0]))
    return sparse_train_metadata, dense_train_metadata


def create_metadata(args, trial_id, use_metafeature=False):
    data_folder = 'data/' + args.benchmark + '/'
    sample_ratio = args.sample_ratio
    if args.benchmark == 'svm':
        hp_range = 6
        hp_indicator = 3
    elif args.benchmark == 'weka':
        hp_range = 103
        hp_indicator = 64
    else:
        raise ValueError('Invalid benchmark name!', args.benchmark)

    if not os.path.exists(data_folder):
        print('Invalid data folder!')
        exit(0)

    all_dataset_name = sorted(os.listdir(data_folder))
    test_dataset_index = trial_id % len(all_dataset_name)

    train_metadata = list()
    test_metadata = None
    for i, item in enumerate(all_dataset_name):
        if i != test_dataset_index:
            train_metadata.append(read_metadata_file(data_folder+item))
        else:
            test_metadata = read_metadata_file(data_folder + item)
            if not use_metafeature:
                test_metadata = test_metadata[:, 0:hp_range + 1]
            if args.benchmark == 'weka':
                meta_size = test_metadata.shape[0]
                # Keep each method at the same trial with same test metadata.
                np.random.seed(trial_id)
                indexes = list(range(meta_size))
                np.random.shuffle(indexes)
                indexes = indexes[: 1000]
                test_metadata = test_metadata[indexes]
    assert test_metadata is not None

    train_metadata, dense_train_metadata = sparse_extraction(
        train_metadata, hp_range, hp_indicator, metafeature=use_metafeature)
    if sample_ratio > 0 and sample_ratio != 1.:
        train_metadata = sample_metadata(dense_train_metadata, trial_id, proportion=sample_ratio)
        # print('The number of metadata in each trials', len(train_metadata[0]))
    return train_metadata, test_metadata
