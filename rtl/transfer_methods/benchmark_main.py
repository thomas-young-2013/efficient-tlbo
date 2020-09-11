import sys
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--num_rep', type=int, default=50)
parser.add_argument('--sample_ratio', type=float, default=1.)
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--benchmark', choices=['svm', 'resnet', 'weka', 'xgb'], default='svm')
parser.add_argument('--use_tex', type=bool, default=False)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--algos', type=int, default=-1)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--num_init', type=int, default=3)
parser.add_argument('--algos_str', type=str)
args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/RTL')
if args.mode == 'server':
    sys.path.append('/home/liyang/codes/RTS')

from rtl.acquisition_function.ei import EI
from rtl.transfer_methods.benchmark_smbo import SMBO
from rtl.facade.two_stage_surrogate import TwoStageSurrogate
from rtl.facade.ranking_weighted_ensemble_surrogate import RankingWeightedEnsembleSurrogate
from rtl.facade.robust_ensemble_surrogate import RobustEnsembleSurrogate
from rtl.facade.gp_stacking_surrogate import GPStackingSurrogate
from rtl.facade.independent_surrogate import IndependentSurrogate
from rtl.facade.scot_surrogate import SCoTSurrogate
from rtl.facade.mklgp_surrogate import MKLGPSurrogate
from rtl.facade.robust_transfer_learning_surrogate import RobustTLSurrogate
from rtl.facade.transfer_surrogate import TransferSurrogate
from rtl.facade.owesome_surrogate import OwesomeSurrogate
from rtl.facade.great_surrogate import GreatSurrogate
from rtl.facade.nn_smfo_surrogate import NNSMFO
from rtl.facade.random_surrogate import RandomSearch
from rtl.facade.pogpe_surrogate import POGPESurrogate

from rtl.utils.metadata_extract import create_metadata
from rtl.utils.load_resnet_metadata import load_resnet_metadata
from rtl.utils.load_xgb_metadata import load_xgb_metadata
from rtl.utils.parallel_runner import run_parallel_async

algo_num = 20
benchmark = args.benchmark
frac = args.sample_ratio
steps = args.steps
algo_name = {0: 'TST-R', 1: 'RGPE', 2: 'VIZIER', 3: 'POGPE', 4: 'I-GP', 5: 'I-RF', 6: 'TST-m', 7: 'NN-SMFO',
             8: 'Random Search', 9: 'SCoT', 10: 'RTL-HPO', 11: 'MKL-GP', 12: 'RTL-HPO-ws',
             13: 'Trans-RTL', 14: 'Trans-RTL-plus', 15: 'TEST_HEDGE', 16: 'RTL-HPO'}
runtime_metric_format = './data/%s_res/%s_runtime_metric_%d_%d_%d_%.3f.npy'


def eval_algo(algo_id):
    print('Evaluating algorithm', algo_name[algo_id])
    acc, adtm = dict(), dict()
    p = list()
    # Store the final weights.
    weights = list()
    # Store the cpu time.
    time_tick = list()
    # Store all weights across trials.
    trial_weight = dict()
    for i in range(steps + args.num_init):
        acc[i] = []
        adtm[i] = []

    # Set bandwidth for TST.
    if args.benchmark == 'weka':
        bd = 0.9
    elif args.benchmark == 'svm':
        bd = 0.1
    else:
        bd = 0.9

    def save_intermediate_data():
        # Save the temporary runtime metrics.
        runtime_metric_file = runtime_metric_format % (benchmark, benchmark, algo_id, steps,
                                                       args.num_rep, args.sample_ratio)
        with open(runtime_metric_file, 'wb') as f:
            pickle.dump([acc, adtm, time_tick], f)

    # RTL settings.
    def create_surrogate():
        ts = RobustTLSurrogate(train_metadata, test_metadata)
        if benchmark == 'weka':
            ts.set_hp(2, 0.1)
            ts.set_mu(1.)
        elif benchmark == 'svm':
            ts.set_hp(2, 0.3)
            ts.set_mu(0.)
            ts.set_rank(3)
        elif benchmark == 'resnet':
            ts.set_hp(2, 0)
            ts.set_mu(1.)
            ts.set_rank(3)
        elif benchmark == 'xgb':
            ts.set_hp(2, 1.)
            ts.set_mu(0)
            ts.set_rank(3)
        else:
            raise ValueError('Invalid benchmark!')
        return ts

    for trial_id in range(args.num_rep):
        print('-'*45 + ('Method %s - TRIAL ID: %d' % (algo_name[algo_id], trial_id)) + '-'*45)
        trial_weight[trial_id] = []
        if args.benchmark == 'resnet':
            train_metadata, test_metadata = load_resnet_metadata(trial_id)
        elif args.benchmark == 'xgb':
            train_metadata, test_metadata = load_xgb_metadata(trial_id, ratio=frac)
        elif args.benchmark == 'svm':
            if algo_id in [11, 9, 6, 3]:
                train_metadata, test_metadata = create_metadata(args, trial_id, use_metafeature=True)
            else:
                train_metadata, test_metadata = create_metadata(args, trial_id)

        if algo_id == 0:
            # TST-R
            ts = TwoStageSurrogate(train_metadata, test_metadata, bandwidth=bd)
        elif algo_id == 1:
            # RGPE
            ts = RankingWeightedEnsembleSurrogate(train_metadata, test_metadata)
        elif algo_id == 2:
            # Vizier
            ts = GPStackingSurrogate(train_metadata, test_metadata)
        elif algo_id == 3:
            # POGPE
            ts = POGPESurrogate(train_metadata, test_metadata)
        elif algo_id == 4:
            # I-GP
            ts = IndependentSurrogate(train_metadata, test_metadata)
        elif algo_id == 5:
            # I-RF
            ts = IndependentSurrogate(train_metadata, test_metadata, base_model='rf')
        elif algo_id == 6:
            # TST-m
            if args.benchmark == 'svm':
                ts = TwoStageSurrogate(train_metadata, test_metadata, bandwidth=3.25, use_ranking=False)
            else:
                raise ValueError('No metafeature in this dataset!')
        elif algo_id == 7:
            # Nearest Model Free Surrogate.
            ts = NNSMFO(train_metadata, test_metadata)
        elif algo_id == 8:
            # Random Search.
            ts = RandomSearch(train_metadata, test_metadata)
        elif algo_id == 9:
            # SCoT.
            if args.benchmark == 'svm':
                train_metadata, test_metadata = create_metadata(args, trial_id, use_metafeature=True)
            else:
                raise ValueError('No metafeature in this dataset!')
            ts = SCoTSurrogate(train_metadata, test_metadata)
        elif algo_id == 10:
            print('rtl-hpo without warm starting', trial_id)
            ts = create_surrogate()
        elif algo_id == 12:
            print('rtl-hpo with warm starting', trial_id)
            ts = create_surrogate()
        elif algo_id == 11:
            # MKL-GP
            if args.benchmark == 'svm':
                train_metadata, test_metadata = create_metadata(args, trial_id, use_metafeature=True)
            else:
                raise ValueError('No metafeature in this dataset!')
            ts = MKLGPSurrogate(train_metadata, test_metadata)
        elif algo_id == 13:
            ts = TransferSurrogate(train_metadata, test_metadata, fusion=False, loss_type=3, lbd=0.)
        elif algo_id == 14:
            ts = TransferSurrogate(train_metadata, test_metadata, fusion=True, loss_type=3, alpha=1.)
        elif algo_id == 15:
            # Test hedge algorithm.
            ts = GreatSurrogate(train_metadata, test_metadata, use_hedge=True)
            ts.init = 1
            ts.set_lbd(True, 1)
            ts.set_params(1, 0.1)
        elif algo_id == 16:
            # RTL-HPO
            ts = GreatSurrogate(train_metadata, test_metadata)
            ts.init = 1
        else:
            raise ValueError('Invalid transfer learning methods!', algo_id)
        ts.set_mode(args.debug)
        acquisition_func = EI(ts, par=0.01)
        smbo = SMBO(test_metadata, acquisition_func, ts, num_initial_design=args.num_init, random_seed=trial_id)
        for i in range(steps):
            smbo.iterate()

            # If initialize with warm starts, add the warm start performance data.
            disp_ws = False
            if disp_ws:
                index = i + (0 if ts.init == -1 else args.num_init)
                if i == 0 and ts.init != -1:
                    acc_perf, err_perf = smbo.get_initial_perf()
                    for j in range(args.num_init):
                        adtm[j].append(err_perf[j])
                        acc[j].append(acc_perf[j])
            else:
                index = i
            if algo_id == 15:
                if i == 0:
                    p.append([])
                p[-1].append(ts.get_p()[0])
            adtm[index].append(smbo.get_relative_error())
            acc[index].append(smbo.get_best_accuracy())
            if args.debug:
                print('='*20, smbo.get_best_accuracy(), smbo.get_relative_error())
            if hasattr(ts, 'get_weights'):
                trial_weight[trial_id].append(ts.get_weights())
            if hasattr(ts, 'get_weights') and i == steps - 1:
                weights.append(ts.get_weights())
            if hasattr(ts, 'get_weights') and args.debug:
                tmp_w = ts.get_weights()
                print('|%.3f ' * len(tmp_w) % tuple(tmp_w))
            if algo_id in [3, 9]:
                save_intermediate_data()
        time_tick.append(smbo.get_time_cost()[args.num_init:])

        if algo_id in [3, 9, 11]:
            # Save the temporary runtime metrics.
            runtime_metric_file = runtime_metric_format % (
            benchmark, benchmark, algo_id, steps, args.num_rep, args.sample_ratio)
            with open(runtime_metric_file, 'wb') as f:
                pickle.dump([acc, adtm, time_tick], f)

    # In debug mode, save the weights for basic surrogates.
    if args.debug and algo_id in [0, 7, 10]:
        # Display the weights for each source.
        weights = np.array(weights)
        print('Mean weight for each source', weights.mean(axis=0))
        np.save('data/%s_res/weights/weights_%s_%d_%d_%d.npy' % (benchmark, benchmark, algo_id, args.num_rep, steps), weights)

        all_weights = list()
        for i in range(args.num_rep):
            all_weights.append(trial_weight[i])
        all_weights = np.array(all_weights)
        np.save('data/%s_res/weights/weights_%s_%d_%d_%d_all.npy' % (benchmark, benchmark, algo_id, args.num_rep, steps), all_weights)

    # Print the metrics.
    for i in range(steps):
        metric = [np.mean(acc[i]), np.std(acc[i]), np.mean(adtm[i]), np.std(adtm[i])]
        print('{0}: acc mean|acc std|adtm mean|adtm std ==> {1}'.format(algo_name[algo_id], str(metric)))

    # Save all runtime metrics.
    runtime_metric_file = runtime_metric_format % (benchmark, benchmark, algo_id, steps, args.num_rep, args.sample_ratio)
    with open(runtime_metric_file, 'wb') as f:
        pickle.dump([acc, adtm, time_tick], f)
    if algo_id == 15:
        p = np.array(p)
        print('Mean weight for each source', p.mean(axis=0))
        np.save('data/%s_res/weights/p_%s_%d_%d_%d.npy' % (benchmark, benchmark, algo_id, args.num_rep, steps),
                p)


if __name__ == "__main__":
    pool_size = args.batch
    if args.algos == -1:
        run_parallel_async(eval_algo, list(range(algo_num)), pool_size=pool_size)
    elif args.algos == -3:
        configs = list(map(lambda x: int(x), args.algos_str.split(',')))
        run_parallel_async(eval_algo, configs, pool_size=pool_size)
    elif args.algos < algo_num:
        eval_algo(args.algos)
    else:
        raise ValueError('Invalid algorithm ID!', args.algos)
