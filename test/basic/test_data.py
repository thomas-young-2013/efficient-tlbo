import os
import re
import sys
import time
import pickle
import argparse
import numpy as np

sys.path.insert(0, '.')
from tlbo.facade.random_surrogate import RandomSearch
from tlbo.framework.smbo_offline import SMBO_OFFLINE
from tlbo.framework.smbo_sst import SMBO_SEARCH_SPACE_TRANSFER
from tlbo.config_space.space_instance import get_configspace_instance

sys.path.insert(0, './test')
from basic.utils import color_str

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, default='main')
parser.add_argument('--exp_id', type=str, default='main')
# parser.add_argument('--algo_id', type=str, default='random_forest')
parser.add_argument('--methods', type=str, default='rgpe')
parser.add_argument('--surrogate_type', type=str, default='rf')
parser.add_argument('--test_mode', type=str, default='random')
parser.add_argument('--trial_num', type=int, default=50)
parser.add_argument('--init_num', type=int, default=0)
parser.add_argument('--run_num', type=int, default=-1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_source_data', type=int, default=50)
parser.add_argument('--num_source_problem', type=int, default=-1)
parser.add_argument('--num_target_data', type=int, default=10000)
parser.add_argument('--num_random_data', type=int, default=20000)
parser.add_argument('--save_weight', type=str, default='false')
args = parser.parse_args()
# algo_id = args.algo_id
exp_id = args.exp_id
task_id = args.task_id
surrogate_type = args.surrogate_type
n_src_data = args.num_source_data
num_source_problem = args.num_source_problem
n_target_data = args.num_target_data
num_random_data = args.num_random_data
trial_num = args.trial_num
init_num = args.init_num
seed = args.seed
run_num = args.run_num
test_mode = args.test_mode
save_weight = args.save_weight
baselines = args.methods.split(',')

data_dir = 'data/hpo_data/'
assert test_mode in ['bo', 'random']
if init_num > 0:
    enable_init_design = True
else:
    enable_init_design = False
    # Default number of random configurations.
    init_num = 3

# algorithms = ['lightgbm', 'random_forest', 'linear', 'adaboost', 'lda', 'extra_trees']
algorithms = ['lightgbm', 'random_forest', 'linear', 'adaboost', 'extra_trees']
algo_str = '|'.join(algorithms)
pattern = '(.*)-(%s)-(\d+).pkl' % algo_str


def load_hpo_history(algo_id):
    source_hpo_ids, source_hpo_data = list(), list()
    random_hpo_data = list()
    for _file in sorted(os.listdir(data_dir)):
        if _file.endswith('.pkl') and _file.find(algo_id) != -1:
            result = re.search(pattern, _file, re.I)
            if result is None:
                continue
            dataset_id, algo_name, total_trial_num = result.group(1), result.group(2), result.group(3)
            if int(total_trial_num) != n_target_data:
                continue
            with open(data_dir + _file, 'rb') as f:
                data = pickle.load(f)
                perfs = np.array(list(data.values()))
            p_max, p_min = np.max(perfs), np.min(perfs)
            if p_max == p_min:
                continue
            if (perfs == perfs[0]).all():
                continue
            if test_mode == 'random':
                _file = data_dir + '%s-%s-random-%d.pkl' % (dataset_id, algo_id, num_random_data)
                if not os.path.exists(_file):
                    continue
            source_hpo_ids.append(dataset_id)
            source_hpo_data.append(data)
            print(len(data), dataset_id, algo_name)
    assert len(source_hpo_ids) == len(source_hpo_data)
    print('Load %s source hpo problems for algorithm %s.' % (len(source_hpo_ids), algo_id))

    # Load random hpo data to test the transfer performance.
    if test_mode == 'random':
        for id, hpo_id in enumerate(source_hpo_ids):
            _file = data_dir + '%s-%s-random-%d.pkl' % (hpo_id, algo_id, num_random_data)
            with open(_file, 'rb') as f:
                data = pickle.load(f)
                perfs = np.array(list(data.values()))
                p_max, p_min = np.max(perfs), np.min(perfs)
                if p_max == p_min:
                    print('The same perfs found in the %d-th problem' % id)
                    data = source_hpo_data[id].copy()
                random_hpo_data.append(data)
                print('load random', len(data))

    print('Load meta-features for each dataset.')
    meta_features = list()
    with open(data_dir + 'dataset_metafeatures.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
        dataset_ids = [item for item in dataset_info['task_ids']]
        dataset_meta_features = list(dataset_info['dataset_embedding'])
        meta_features_dict = dict(zip(dataset_ids, dataset_meta_features))
    for hpo_id in source_hpo_ids:
        assert hpo_id in dataset_ids
        meta_features.append(np.array(meta_features_dict[hpo_id], dtype=np.float64))
    return source_hpo_ids, source_hpo_data, random_hpo_data, meta_features


perf_file = 'logs/perf_file.pkl'
if __name__ == "__main__":
    if os.path.exists(perf_file):
        with open(perf_file, 'rb') as f:
            perf_stat = pickle.load(f)
            print('load stat')
    else:
        perf_stat = dict()
        for algo_id in algorithms:
            hpo_ids, hpo_data, random_test_data, meta_features = load_hpo_history(algo_id)
            for data_name, data in zip(hpo_ids, hpo_data):
                if data_name not in perf_stat.keys():
                    perf_stat[data_name] = dict()
                perfs = np.array(list(data.values()))
                perf_stat[data_name][algo_id] = perfs
        with open(perf_file, 'wb') as f:
            pickle.dump(perf_stat, f)
            print('dump stat')

    stat = dict()
    result = {
        '<0.001': [],
        '0.001 ~ 0.05': [],
        '0.05 ~ 0.1': [],
        '>0.1': [],
    }
    for k1, v1 in perf_stat.items():
        if k1 not in stat.keys():
            stat[k1] = dict()
        for k2, perfs in v1.items():
            perfs = perfs[:100]  # first 100 trials in BO
            p_max = np.max(perfs)
            p_min = np.min(perfs)
            p_median = np.median(perfs)
            p_first = perfs[0]

            # gap = p_first - p_min
            gap = p_median - p_min

            name = k1 + ' | ' + k2
            # name = k1 + '_' + k2
            if gap < 0.001:
                color = 'red'
                result['<0.001'].append(name)
            elif gap < 0.02:
                color = 'yellow'
                result['0.001 ~ 0.02'].append(name)
            elif gap < 0.1:
                color = 'blue'
                result['0.02 ~ 0.1'].append(name)
            else:
                color = None
                result['>0.1'].append(name)

            s = "%.6f" % gap
            if color is not None:
                s = color_str(s, color)
            stat[k1][k2] = s

    # import pprint
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(result)
    print(result['>0.1'])

    from prettytable import PrettyTable
    head = [' '] + algorithms
    table = PrettyTable(head)
    for data_name in stat.keys():
        row = [data_name]
        for algo_id in algorithms:
            item = stat[data_name][algo_id]
            row.append(item)
        table.add_row(row)
    print(table)
