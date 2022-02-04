import os

NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = NUM_THREADS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS  # export NUMEXPR_NUM_THREADS=1

import re
import sys
import time
import pickle
import argparse
import numpy as np
from functools import partial
from tqdm import tqdm, trange

sys.path.insert(0, '.')
from tlbo.facade.notl import NoTL
from tlbo.facade.rgpe import RGPE
from tlbo.facade.obtl_es import ES
from tlbo.facade.obtl import OBTL
from tlbo.facade.tst import TST
from tlbo.facade.tstm import TSTM
from tlbo.facade.pogpe import POGPE
from tlbo.facade.stacking_gpr import SGPR
from tlbo.facade.scot import SCoT
from tlbo.facade.mklgp import MKLGP
from tlbo.facade.topo_variant1 import OBTLV
from tlbo.facade.topo_variant2 import TOPO
from tlbo.facade.topo_variant3 import TOPO_V3
from tlbo.facade.topo import TransBO_RGPE
from tlbo.facade.mfes import MFES
from tlbo.facade.rgpe_space import RGPESPACE
from tlbo.facade.tst_space import TSTSPACE
from tlbo.facade.norm import NORM
from tlbo.facade.norm_minus import NORMMinus
from tlbo.facade.norm_tst import NORMTST
from tlbo.facade.random_surrogate import RandomSearch
from tlbo.framework.smbo_offline import SMBO_OFFLINE
from tlbo.framework.smbo_sst import SMBO_SEARCH_SPACE_TRANSFER
from tlbo.framework.smbo_baseline import SMBO_SEARCH_SPACE_Enlarge
from tlbo.config_space.space_instance import get_configspace_instance

from tools.utils import seeds

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, default='main')
parser.add_argument('--exp_id', type=str, default='main')
parser.add_argument('--algo_id', type=str, default='random_forest')
parser.add_argument('--methods', type=str, default='rgpe')
parser.add_argument('--surrogate_type', type=str, default='gp')
parser.add_argument('--test_mode', type=str, default='random')
parser.add_argument('--trial_num', type=int, default=50)
parser.add_argument('--init_num', type=int, default=0)
# parser.add_argument('--run_num', type=int, default=-1)
parser.add_argument('--num_source_trial', type=int, default=50)
parser.add_argument('--num_source_problem', type=int, default=-1)
parser.add_argument('--task_set', type=str, default='class1', choices=['class1', 'class2', 'full'])
parser.add_argument('--target_set', type=str, default='class1')
parser.add_argument('--num_source_data', type=int, default=10000)
parser.add_argument('--num_random_data', type=int, default=50000)
parser.add_argument('--save_weight', type=str, default='false')
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

default_pmin, default_pmax = 5, 50
parser.add_argument('--pmin', type=int, default=default_pmin)
parser.add_argument('--pmax', type=int, default=default_pmax)
args = parser.parse_args()

algo_id = args.algo_id
exp_id = args.exp_id
task_id = args.task_id
task_set = args.task_set
targets = args.target_set
surrogate_type = args.surrogate_type
n_src_trial = args.num_source_trial
num_source_problem = args.num_source_problem
n_source_data = args.num_source_data
num_random_data = args.num_random_data
trial_num = args.trial_num
init_num = args.init_num
# run_num = args.run_num
test_mode = args.test_mode
save_weight = args.save_weight
baselines = args.methods.split(',')
rep = args.rep
start_id = args.start_id

pmin = args.pmin
pmax = args.pmax

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
src_pattern = '(.*)-(%s)-(\d+).pkl' % algo_str


def get_data_set(set_name):
    assert set_name in ['class1', 'class2', 'full']
    if set_name == 'class1':
        data_set = ['kc1', 'pollen', 'madelon', 'winequality_white', 'sick']
    elif set_name == 'class2':
        data_set = ['kc1', 'pollen', 'madelon', 'winequality_white', 'sick',
                    'quake', 'hypothyroid(1)', 'musk', 'page-blocks(1)', 'page-blocks(2)',
                    'satimage', 'segment', 'waveform-5000(2)']
    else:
        data_set = ['kc1', 'pollen', 'madelon', 'winequality_white', 'sick',
                    'quake', 'hypothyroid(1)', 'musk', 'page-blocks(1)', 'page-blocks(2)',
                    'satimage', 'segment', 'waveform-5000(2)',
                    'space_ga', 'splice', 'kr-vs-kp', 'hypothyroid(2)', 'spambase', 'analcatdata_supreme', 'balloon',
                    'cpu_act', 'cpu_small', 'bank32nh', 'puma8NH', 'wind', 'mushroom', 'waveform-5000(1)',
                    'delta_ailerons', 'abalone', 'optdigits']
    return data_set


def load_hpo_history():
    source_hpo_ids, source_hpo_data = list(), list()
    random_hpo_data = list()
    for _file in tqdm(sorted(os.listdir(data_dir))):
        if _file.endswith('.pkl') and _file.find(algo_id) != -1:
            result = re.search(src_pattern, _file, re.I)
            if result is None:
                continue
            dataset_id, algo_name, total_trial_num = result.group(1), result.group(2), result.group(3)
            if int(total_trial_num) != n_source_data:
                continue
            with open(os.path.join(data_dir, _file), 'rb') as f:
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

            # TODO: Add test perf
            if test_mode == 'bo':
                raise NotImplementedError('TODO: Add test perf')
            if perfs.ndim == 2:
                assert perfs.shape[1] == 2
                _data = {k: v[0] for k, v in data.items()}
            else:
                _data = data
            source_hpo_data.append(_data)

    assert len(source_hpo_ids) == len(source_hpo_data)
    print('Load %s source hpo problems for algorithm %s.' % (len(source_hpo_ids), algo_id))

    # Load random hpo data to test the transfer performance.
    if test_mode == 'random':
        for id, hpo_id in tqdm(list(enumerate(source_hpo_ids))):
            _file = data_dir + '%s-%s-random-%d.pkl' % (hpo_id, algo_id, num_random_data)
            with open(_file, 'rb') as f:
                data = pickle.load(f)
                perfs = np.array(list(data.values()))
                p_max, p_min = np.max(perfs), np.min(perfs)
                if p_max == p_min:
                    raise ValueError('The same perfs found in the %d-th problem' % id)
                    # data = source_hpo_data[id].copy()
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


def extract_data(task_set):
    if task_set == 'full':
        hpo_ids, hpo_data, random_test_data, meta_features = load_hpo_history()
    elif task_set in ['class1', 'class2']:
        dataset_ids = get_data_set(task_set)

        hpo_ids, hpo_data, random_test_data, meta_features = list(), list(), list(), list()
        hpo_ids_, hpo_data_, random_test_data_, meta_features_ = load_hpo_history()
        for _idx, _id in enumerate(hpo_ids_):
            if _id in dataset_ids:
                hpo_ids.append(hpo_ids_[_idx])
                hpo_data.append(hpo_data_[_idx])
                random_test_data.append(random_test_data_[_idx])
                meta_features.append(meta_features_[_idx])
    else:
        raise ValueError('Invalid Task Set.')
    return hpo_ids, hpo_data, random_test_data, meta_features


if __name__ == "__main__":
    hpo_ids, hpo_data, random_test_data, meta_features = extract_data(task_set)
    algo_name = 'liblinear_svc' if algo_id == 'linear' else algo_id
    config_space = get_configspace_instance(algo_id=algo_name)
    num_source_problem = (len(hpo_ids) - 1) if num_source_problem == -1 else num_source_problem
    # if 'rs' in baselines and len(random_test_data) == 0:
    #     raise ValueError('The random test data is empty!')

    run_id = list()
    if targets in ['class1', 'class2', 'full']:
        targets = get_data_set(targets)
    else:
        targets = targets.split(',')
    for target_id in targets:
        target_idx = hpo_ids.index(target_id)
        run_id.append(target_idx)

    # Exp folder to save results.
    exp_dir = 'data/exp_results/%s_%s_%s_%d_%d/' % (exp_id, test_mode, task_set, num_source_problem, num_random_data)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    pbar = tqdm(total=rep * len(baselines) * len(run_id) * trial_num)
    for id in run_id:
        for mth in baselines:
            for rep_id in range(start_id, start_id + rep):
                seed = seeds[rep_id]
                # exp_results = list()
                # target_weights = list()
                print('=== start rep', rep_id, 'seed', seed)

                print('=' * 20)
                print('[%s-%s] Evaluate %d-th problem - %s[%d].' % (algo_id, mth, id + 1, hpo_ids[id], rep_id))
                pbar.set_description('[%s-%s] %d-th - %s[%d]' % (algo_id, mth, id + 1, hpo_ids[id], rep_id))
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
                elif mth == 'ultra':
                    surrogate_class = RGPE
                elif mth.startswith('rgpe-space'):
                    surrogate_class = RGPESPACE
                elif mth.startswith('tst-space'):
                    surrogate_class = TSTSPACE
                elif 'tst' in mth:
                    surrogate_class = NORMTST
                elif mth.endswith('-'):
                    surrogate_class = NORMMinus
                elif 'space' in mth:
                    surrogate_class = NORM
                elif mth in ['box', 'ellipsoid']:
                    surrogate_class = NoTL
                elif mth in ['rgpe-box', 'rgpe-ellipsoid']:
                    surrogate_class = RGPE
                else:
                    raise ValueError('Invalid baseline name - %s.' % mth)
                if mth not in ['mklgp', 'scot', 'tstm']:
                    surrogate = surrogate_class(config_space, source_hpo_data, target_hpo_data, seed,
                                                surrogate_type=surrogate_type,
                                                num_src_hpo_trial=n_src_trial)
                else:
                    surrogate = surrogate_class(config_space, source_hpo_data, target_hpo_data, seed,
                                                surrogate_type=surrogate_type,
                                                num_src_hpo_trial=n_src_trial, metafeatures=dataset_meta_features)

                if '-dif' in mth:
                    surrogate.same = False
                if mth.endswith('-i') or '-i-' in mth:
                    surrogate.increasing_weight = True
                elif mth.endswith('-d') or '-d-' in mth:
                    surrogate.nondecreasing_weight = True

                if mth == "ultra":
                    smbo_framework = SMBO_SEARCH_SPACE_TRANSFER

                if 'rf' in mth:
                    model = 'rf'
                elif 'knn' in mth:
                    model = 'knn'
                elif 'gp' in mth:
                    model = 'gp'
                else:
                    model = 'svm'

                if 'all+-sample+' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='all+-sample+', model=model)
                elif 'all+-sample' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='all+-sample', model=model)
                elif 'all+-threshold' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='all+-threshold', model=model)
                elif 'all+' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='all+', model=model)
                elif 'all' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='all', model=model)
                elif 'sample-new' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='sample-new', model=model)
                elif 'sample' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='sample', model=model)
                elif 'space' in mth:
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='best', model=model)
                elif mth == 'box':
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='box', model=model)
                elif mth == 'ellipsoid':
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='ellipsoid', model=model)
                elif mth == 'rgpe-box':
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='box', model=model)
                elif mth == 'rgpe-ellipsoid':
                    smbo_framework = partial(SMBO_SEARCH_SPACE_Enlarge, mode='ellipsoid', model=model)
                else:
                    smbo_framework = SMBO_OFFLINE

                smbo = smbo_framework(target_hpo_data, config_space, surrogate,
                                      random_seed=seed, max_runs=trial_num,
                                      source_hpo_data=source_hpo_data,
                                      num_src_hpo_trial=n_src_trial,
                                      surrogate_type=surrogate_type,
                                      enable_init_design=enable_init_design,
                                      initial_runs=init_num,
                                      acq_func='ei')

                if hasattr(smbo, 'p_min'):
                    smbo.p_min = pmin
                    smbo.p_max = pmax
                    print('use pmin/max:', smbo.p_min, smbo.p_max)
                if 'v2' in mth:
                    smbo.use_correct_rate = True

                result = list()
                for _iter_id in range(trial_num):
                    config, _, perf, _ = smbo.iterate()
                    time_taken = time.time() - start_time
                    adtm, y_inc = smbo.get_adtm(), smbo.get_inc_y()
                    result.append([adtm, y_inc, time_taken])
                    pbar.update(1)
                # exp_results.append(result)
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

                # target_weights.append(surrogate.target_weight)

                # Save the running results on the fly with overwriting.
                # if run_num == len(hpo_ids):
                #     if pmin != default_pmin or pmax != default_pmax:
                #         mth_file = '%s_%d_%d_%s_%d_%d_%s_%s_%d.pkl' % (
                #             mth, pmin, pmax, algo_id, n_src_trial, trial_num, surrogate_type, task_id, seed)
                #     else:
                #         mth_file = '%s_%s_%d_%d_%s_%s_%d.pkl' % (
                #             mth, algo_id, n_src_trial, trial_num, surrogate_type, task_id, seed)
                #     with open(exp_dir + mth_file, 'wb') as f:
                #         data = [np.array(exp_results), np.mean(exp_results, axis=0)]
                #         pickle.dump(data, f)
                #
                #     if save_weight == 'true':
                #         mth_file = 'w_%s_%s_%d_%d_%s_%s_%d.pkl' % (
                #             mth, algo_id, n_src_trial, trial_num, surrogate_type, task_id, seed)
                #         with open(exp_dir + mth_file, 'wb') as f:
                #             data = target_weights
                #             pickle.dump(data, f)

                if pmin != default_pmin or pmax != default_pmax:
                    mth_file = '%s_%d_%d_%s_%s_%d_%d_%s_%s_%d.pkl' % (
                        mth, pmin, pmax, hpo_ids[id], algo_id, n_src_trial, trial_num, surrogate_type, task_id, seed)
                else:
                    mth_file = '%s_%s_%s_%d_%d_%s_%s_%d.pkl' % (
                        mth, hpo_ids[id], algo_id, n_src_trial, trial_num, surrogate_type, task_id, seed)
                with open(exp_dir + mth_file, 'wb') as f:
                    data = np.array(result)
                    pickle.dump(data, f)

                if save_weight == 'true':
                    mth_file = 'w_%s_%s_%s_%d_%d_%s_%s_%d.pkl' % (
                        mth, hpo_ids[id], algo_id, n_src_trial, trial_num, surrogate_type, task_id, seed)
                    with open(exp_dir + mth_file, 'wb') as f:
                        data = surrogate.target_weight
                        pickle.dump(data, f)
    pbar.close()
