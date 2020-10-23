import os
import re
import sys
import time
import pickle
import argparse
import numpy as np

sys.path.append(os.getcwd())
from tlbo.framework.smbo_offline import SMBO_OFFLINE
from tlbo.facade.notl import NoTL
from tlbo.facade.rgpe import RGPE
from tlbo.facade.obtl_es import ES
from tlbo.facade.obtl import OBTL
from tlbo.facade.random_surrogate import RandomSearch
from tlbo.facade.tst import TST
from tlbo.facade.tstm import TSTM
from tlbo.facade.pogpe import POGPE
from tlbo.facade.stacking_gpr import SGPR
from tlbo.facade.scot import SCoT
from tlbo.facade.mklgp import MKLGP
from tlbo.facade.topo_variant1 import OBTLV
from tlbo.facade.topo_variant2 import TOPO
from tlbo.facade.topo_variant3 import TOPO_V3
from tlbo.config_space.space_instance import get_configspace_instance

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, default='main')
parser.add_argument('--exp_id', type=str, default='main')
parser.add_argument('--algo_id', type=str, default='random_forest')
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
algo_id = args.algo_id
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

algorithms = ['lightgbm', 'random_forest', 'linear', 'adaboost', 'lda', 'extra_trees']
algo_str = '|'.join(algorithms)
pattern = '(.*)-(%s)-(\d+).pkl' % algo_str


def load_hpo_history():
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


if __name__ == "__main__":
    hpo_ids, hpo_data, random_test_data, meta_features = load_hpo_history()
    algo_name = 'liblinear_svc' if algo_id == 'linear' else algo_id
    config_space = get_configspace_instance(algo_id=algo_name)
    np.random.seed(seed)
    seeds = np.random.randint(low=1, high=10000, size=len(hpo_ids))
    run_num = len(hpo_ids) if run_num == -1 else run_num
    num_source_problem = (len(hpo_ids) - 1) if num_source_problem == -1 else num_source_problem
    # if 'rs' in baselines and len(random_test_data) == 0:
    #     raise ValueError('The random test data is empty!')

    # Exp folder to save results.
    exp_dir = 'data/exp_results/%s_%s_%d_%d/' % (exp_id, test_mode, num_source_problem, num_random_data)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    target_weights = []

    for mth in baselines:
        exp_results = list()
        for id in range(run_num):
            print('=' * 20)
            print('[%s-%s] Evaluate %d-th problem - %s.' % (algo_id, mth, id + 1, hpo_ids[id]))
            start_time = time.time()

            # Generate the source and target hpo data.
            source_hpo_data, dataset_meta_features = list(), list()
            if test_mode == 'bo':
                target_hpo_data = hpo_data[id]
            else:
                target_hpo_data = random_test_data[id]
            for _id, data in enumerate(hpo_data):
                if _id != id:
                    source_hpo_data.append(data)
                    dataset_meta_features.append(meta_features[_id])

            # Random seed.
            seed = seeds[id]
            # Select a subset of source problems to transfer.
            rng = np.random.RandomState(seed)
            shuffled_ids = np.arange(len(source_hpo_data))
            rng.shuffle(shuffled_ids)
            source_hpo_data = [source_hpo_data[id] for id in shuffled_ids[:num_source_problem]]
            dataset_meta_features = [dataset_meta_features[id] for id in shuffled_ids[:num_source_problem]]
            # Add the meta-features in the target problem.
            dataset_meta_features.append(meta_features[id])

            if mth == 'rgpe':
                surrogate_class = RGPE
            elif mth == 'notl':
                surrogate_class = NoTL
            elif mth == 'es':
                surrogate_class = ES
            elif mth == 'obtl':
                surrogate_class = OBTL
            elif mth == 'obtlv':
                surrogate_class = OBTLV
            elif mth == 'tst':
                surrogate_class = TST
            elif mth == 'pogpe':
                surrogate_class = POGPE
            elif mth == 'sgpr':
                surrogate_class = SGPR
            elif mth == 'scot':
                surrogate_class = SCoT
            elif mth == 'mklgp':
                surrogate_class = MKLGP
            elif mth == 'rs':
                surrogate_class = RandomSearch
            elif mth == 'tstm':
                surrogate_class = TSTM
            elif mth == 'topo':
                surrogate_class = OBTLV
            elif mth == 'topo_v3':
                surrogate_class = TOPO_V3
            else:
                raise ValueError('Invalid baseline name - %s.' % mth)
            if mth not in ['mklgp', 'scot', 'tstm']:
                surrogate = surrogate_class(config_space, source_hpo_data, target_hpo_data, seed,
                                            surrogate_type=surrogate_type,
                                            num_src_hpo_trial=n_src_data)
            else:
                surrogate = surrogate_class(config_space, source_hpo_data, target_hpo_data, seed,
                                            surrogate_type=surrogate_type,
                                            num_src_hpo_trial=n_src_data, metafeatures=dataset_meta_features)

            smbo = SMBO_OFFLINE(target_hpo_data, config_space, surrogate,
                                random_seed=seed, max_runs=trial_num,
                                source_hpo_data=source_hpo_data,
                                num_src_hpo_trial=n_src_data,
                                surrogate_type=surrogate_type,
                                enable_init_design=enable_init_design,
                                initial_runs=init_num,
                                acq_func='ei')

            result = list()
            rnd_target_perfs = [_perf for (_, _perf) in list(random_test_data[id].items())]
            rnd_ymax, rnd_ymin = np.max(rnd_target_perfs), np.min(rnd_target_perfs)

            for _iter_id in range(trial_num):
                if surrogate.method_id == 'rs':
                    _perfs = rnd_target_perfs[:(_iter_id + 1)]
                    y_inc = np.min(_perfs)
                    adtm = (y_inc - rnd_ymin) / (rnd_ymax - rnd_ymin)
                    result.append([adtm, y_inc, 0.1])
                else:
                    config, _, perf, _ = smbo.iterate()
                    time_taken = time.time() - start_time
                    adtm, y_inc = smbo.get_adtm(), smbo.get_inc_y()
                    result.append([adtm, y_inc, time_taken])
            exp_results.append(result)
            print('In %d-th problem: %s' % (id, hpo_ids[id]), 'adtm, y_inc', result[-1])
            print('min/max', smbo.y_min, smbo.y_max)
            print('mean,std', np.mean(smbo.ys), np.std(smbo.ys))
            if hasattr(surrogate, 'hist_ws'):
                weights = np.array(surrogate.hist_ws)
                trans = lambda x: ','.join([('%.2f' % item) for item in x])
                weight_str = '\n'.join([trans(item) for item in weights])
                print(weight_str)
                print('Weight stats.')
                print(trans(np.mean(weights, axis=0)))
                source_ids = [item[0] for item in enumerate(list(np.mean(weights, axis=0))) if item[1] >= 1e-2]
                print('Source problems used', source_ids)

            target_weights.append(surrogate.target_weight)

            # Save the running results on the fly with overwriting.
            if run_num == len(hpo_ids):
                mth_file = '%s_%s_%d_%d_%s_%s.pkl' % (mth, algo_id, n_src_data, trial_num, surrogate_type, task_id)
                with open(exp_dir + mth_file, 'wb') as f:
                    data = [np.array(exp_results), np.mean(exp_results, axis=0)]
                    pickle.dump(data, f)

                if save_weight == 'true':
                    mth_file = 'w_%s_%s_%d_%d_%s_%s.pkl' % (
                        mth, algo_id, n_src_data, trial_num, surrogate_type, task_id)
                    with open(exp_dir + mth_file, 'wb') as f:
                        data = target_weights
                        pickle.dump(data, f)
