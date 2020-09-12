import os
import sys
import pickle
import argparse
import numpy as np
from collections import OrderedDict

sys.path.append(os.getcwd())
from tlbo.framework.smbo_offline import SMBO_OFFLINE

parser = argparse.ArgumentParser()
parser.add_argument('--algo_id', type=str, default='random_forest')
parser.add_argument('--trial_num', type=int, default=100)
parser.add_argument('--num_source_data', type=int, default=100)
args = parser.parse_args()
algo_id = args.algo_id
n_src_data = args.num_source_data
data_dir = 'data/hpo_data/'


def load_hpo_history():
    source_hpo_ids = list()
    source_hpo_data = list()
    for _file in os.listdir(data_dir):
        if _file.endswith('.pkl') and _file.find(algo_id) != -1:
            source_hpo_ids.append(_file.split('-')[0])
            with open(data_dir + _file, 'rb') as f:
                data = pickle.load(f)
            # partial_data = OrderedDict(list(data.items())[:n_src_data])
            source_hpo_data.append(data)
    assert len(source_hpo_ids) == len(source_hpo_data)
    print('Load %s source hpo problems for algorithm %s.' % (len(source_hpo_ids), algo_id))
    return source_hpo_ids, source_hpo_data


if __name__ == "__main__":
    hpo_ids, hpo_data = load_hpo_history()
    from tlbo.facade.notl import NoTL
    from tlbo.config_space.space_instance import get_configspace_instance
    config_space = get_configspace_instance(algo_id=algo_id)
    for id in range(1):
        # Generate the source and target hpo data.
        target_hpo_data = hpo_data[id]
        source_hpo_data = list()
        for _id, data in enumerate(hpo_data):
            if _id != id:
                source_hpo_data.append(data)
        rng = np.random.RandomState(1)
        surrogate = NoTL(config_space, source_hpo_data, target_hpo_data, rng)
        smbo = SMBO_OFFLINE(target_hpo_data, config_space, surrogate, max_runs=100)
        for _ in range(10):
            config, _, perf, _ = smbo.iterate()
            # print(config, perf)

