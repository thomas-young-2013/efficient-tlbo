import sys
import os
import time
import argparse
import traceback
import pickle as pkl
from collections import OrderedDict
from tqdm import tqdm, trange
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from cls_model import (
    AdaboostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    KNearestNeighborsClassifier, LDA, LightGBM, LibSVM_SVC, LibLinear_SVC,
    Logistic_Regression, QDA, RandomForest,
)

sys.path.append('../mindware')
from mindware.datasets.utils import load_data


model_map = dict(
    lightgbm=LightGBM,
    random_forest=RandomForest,
)

"""
datasets:
class1: 'kc1', 'pollen', 'madelon', 'winequality_white', 'sick'
class2: 'quake', 'hypothyroid(1)', 'musk', 'page-blocks(1)', 'page-blocks(2)',
        'satimage', 'segment', 'waveform-5000(2)'
"""

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['lightgbm', 'random_forest'])
parser.add_argument('--datasets', type=str)
parser.add_argument('--n', type=int, default=50000)
parser.add_argument('--n_jobs', type=int, default=1)
parser.add_argument('--mode', type=str, default='random')
parser.add_argument('--data_dir', type=str, default='../mindware/')

args = parser.parse_args()

print(' ' * 10 + '=== Options ===')
for k, v in vars(args).items():
    print(' ' * 10 + k + ': ' + str(v))

model_str = args.model
model_class = model_map[model_str]
datasets = args.datasets.split(',')
run_count = args.n
n_jobs = args.n_jobs
mode = args.mode
data_dir = args.data_dir


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset, data_dir=data_dir, datanode_returned=False, preprocess=True, task_type=0)
        except Exception as e:
            print(traceback.format_exc())
            raise ValueError('Dataset - %s load error!' % _dataset)


def objective_function(config):
    config_dict = config.get_dictionary()
    model = model_class(**config_dict, n_jobs=n_jobs, random_state=47)

    model.fit(x_train, y_train)
    y_valid_pred = model.predict(x_valid)
    y_test_pred = model.predict(x_test)

    valid_perf = 1 - balanced_accuracy_score(y_valid, y_valid_pred)
    test_perf = 1 - balanced_accuracy_score(y_test, y_test_pred)
    return valid_perf, test_perf


check_datasets(datasets)
cs = model_class.get_cs()


for dataset in datasets:
    # load dataset
    x, y, _ = load_data(dataset, data_dir=data_dir, datanode_returned=False, preprocess=True, task_type=0)
    # 6:2:2
    x_used, x_test, y_used, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    x_train, x_valid, y_train, y_valid = train_test_split(x_used, y_used, test_size=0.25, stratify=y_used,
                                                          random_state=1)

    # make dirs
    saved_file = 'logs/gen_data/%s-%s-%s-%d.pkl' % (dataset, model_str, mode, run_count)
    print('start', saved_file)
    try:
        dir_path = os.path.dirname(saved_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        pass

    # load history from existing file
    if not os.path.exists(saved_file):
        history = OrderedDict()
        print('no existing history.')
    else:
        with open(saved_file, 'rb') as f:
            history = pkl.load(f)
            print('load history. len:', len(history))
    time.sleep(2)

    remain_count = run_count - len(history)
    fail_cnt = 0
    for i in trange(remain_count, desc='[%s %s]' % (dataset, model_str)):
        config = None
        # sample new config
        for _ in range(5000):
            config = cs.sample_configuration(1)
            if config not in history.keys():
                break
        else:
            print('Sample error. quit!')
            break   # end outer loop

        # evaluate
        try:
            val_result, test_result = objective_function(config)
        except Exception:
            fail_cnt += 1
            print(traceback.format_exc())
            print('evaluation failed %d times!' % (fail_cnt, ))
            val_result, test_result = 1.0, 1.0
            if fail_cnt >= 100:
                break
        print(config, val_result, test_result)

        # save history
        history[config] = [val_result, test_result]
        with open(saved_file, 'wb') as f:
            pkl.dump(history, f)

        if (i + 1) % 10000 == 0:
            bak_file = saved_file + '.bak'
            with open(bak_file, 'wb') as f:
                pkl.dump(history, f)

    print('Final file len:', len(history), saved_file)
