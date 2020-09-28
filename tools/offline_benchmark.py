import os
import re
import sys
import time
import pickle
import argparse
import numpy as np

sys.path.append(os.getcwd())
from tlbo.framework.smbo_offline import SMBO_OFFLINE

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, default='main')
parser.add_argument('--algo_id', type=str, default='random_forest')
parser.add_argument('--methods', type=str, default='rgpe')
parser.add_argument('--surrogate_type', type=str, default='rf')
parser.add_argument('--trial_num', type=int, default=50)
parser.add_argument('--init_num', type=int, default=0)
parser.add_argument('--problem_num', type=int, default=-1)
parser.add_argument('--num_source_data', type=int, default=50)
parser.add_argument('--num_target_data', type=int, default=10000)
args = parser.parse_args()
algo_id = args.algo_id
task_id = args.task_id
surrogate_type = args.surrogate_type
n_src_data = args.num_source_data
n_target_data = args.num_target_data
trial_num = args.trial_num
init_num = args.init_num
problem_num = args.problem_num
baselines = args.methods.split(',')
data_dir = 'data/hpo_data/'
exp_dir = 'data/exp_results/'


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
    for _file in sorted(os.listdir(data_dir)):
        if _file.endswith('.pkl') and _file.find(algo_id) != -1:
            result = re.search(pattern, _file, re.I)
            dataset_id, algo_name, total_trial_num = result.group(1), result.group(2), result.group(3)
            # print(dataset_id, algo_name, total_trial_num)
            if int(total_trial_num) != n_target_data:
                continue
            with open(data_dir + _file, 'rb') as f:
                data = pickle.load(f)
                perfs = np.array(list(data.values()))
            if (perfs == perfs[0]).all():
                continue
            source_hpo_ids.append(dataset_id)
            source_hpo_data.append(data)
    assert len(source_hpo_ids) == len(source_hpo_data)
    print('Load %s source hpo problems for algorithm %s.' % (len(source_hpo_ids), algo_id))
    return source_hpo_ids, source_hpo_data


if __name__ == "__main__":
    hpo_ids, hpo_data = load_hpo_history()
    from tlbo.facade.notl import NoTL
    from tlbo.facade.rgpe import RGPE
    from tlbo.facade.ensemble_selection import ES
    from tlbo.facade.random_surrogate import RandomSearch
    from tlbo.facade.tst import TST
    from tlbo.facade.pogpe import POGPE
    from tlbo.facade.stacking_gpr import SGPR
    from tlbo.config_space.space_instance import get_configspace_instance

    algo_name = 'liblinear_svc' if algo_id == 'linear' else algo_id
    config_space = get_configspace_instance(algo_id=algo_name)
    np.random.seed(42)
    seeds = np.random.randint(low=1, high=10000, size=len(hpo_ids))
    p_num = len(hpo_ids) if problem_num == -1 else problem_num

    for mth in baselines:
        exp_results = list()
        for id in range(p_num):
            print('=' * 20)
            print('[%s-%s] Evaluate %d-th problem - %s.' % (algo_id, mth, id + 1, hpo_ids[id]))
            start_time = time.time()
            # Generate the source and target hpo data.
            target_hpo_data = hpo_data[id]
            source_hpo_data = list()
            for _id, data in enumerate(hpo_data):
                if _id != id:
                    source_hpo_data.append(data)
            seed = seeds[id]
            if mth == 'rgpe':
                surrogate_class = RGPE
            elif mth == 'notl':
                surrogate_class = NoTL
            elif mth == 'es':
                surrogate_class = ES
            elif mth == 'tst':
                surrogate_class = TST
            elif mth == 'pogpe':
                surrogate_class = POGPE
            elif mth == 'sgpr':
                surrogate_class = SGPR
            elif mth == 'rs':
                surrogate_class = RandomSearch
            else:
                raise ValueError('Invalid baseline name - %s.' % mth)
            surrogate = surrogate_class(config_space, source_hpo_data, target_hpo_data, seed,
                                        surrogate_type=surrogate_type,
                                        num_src_hpo_trial=n_src_data)
            smbo = SMBO_OFFLINE(target_hpo_data, config_space, surrogate,
                                random_seed=seed, max_runs=trial_num,
                                source_hpo_data=source_hpo_data,
                                num_src_hpo_trial=n_src_data,
                                surrogate_type=surrogate_type,
                                enable_init_design=enable_init_design,
                                initial_runs=init_num,
                                acq_func='ei')
            result = list()
            for _ in range(trial_num):
                config, _, perf, _ = smbo.iterate()
                # print(config, perf)
                time_taken = time.time() - start_time
                adtm, y_inc = smbo.get_adtm(), smbo.get_inc_y()
                # print('%.3f - %.3f' % (adtm, y_inc))
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

        if problem_num == -1:
            mth_file = '%s_%s_%d_%d_%s_%s.pkl' % (mth, algo_id, n_src_data, trial_num, surrogate_type, task_id)
            with open(exp_dir + mth_file, 'wb') as f:
                data = [np.array(exp_results), np.mean(exp_results, axis=0)]
                pickle.dump(data, f)
