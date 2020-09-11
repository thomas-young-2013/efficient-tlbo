import sys
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--num_rep', type=int, default=50)
parser.add_argument('--sample_ratio', type=float, default=1.)
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--benchmark', choices=['svm', 'resnet', 'weka', 'xgb'], default='resnet')
parser.add_argument('--alpha', type=str, default='1.')
parser.add_argument('--init_lbd', type=str, default='1.')
parser.add_argument('--rate', type=str, default='0.3')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--num_init', type=int, default=3)
args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/RTL')
if args.mode == 'server':
    sys.path.append('/home/liyang/codes/RTS')

from rtl.acquisition_function.ei import EI
from rtl.transfer_methods.benchmark_smbo import SMBO
from rtl.facade.great_surrogate import GreatSurrogate
from rtl.utils.metadata_extract import create_metadata
from rtl.utils.load_resnet_metadata import load_resnet_metadata
from rtl.utils.load_xgb_metadata import load_xgb_metadata
from rtl.utils.parallel_runner import run_parallel_async

benchmark = args.benchmark
frac = args.sample_ratio
steps = args.steps
metric_format = './data/%s_res/DEBUG-%s_step-%d_rep-%d_sr_%.3f_alpha-%.3f_lbd-%.3f_rate-%.3f.npy'


def eval_algo(params):
    alpha, init_lbd, rate = params
    print('| %s ==> alpha | init_lambda | reduction_rate |: %s |' % (benchmark, str(params)))

    acc, adtm = dict(), dict()
    for i in range(steps + args.num_init):
        acc[i] = []
        adtm[i] = []

    for trial_id in range(args.num_rep):
        print('-'*10 + ('TRIAL ID: %d' % trial_id) + '-'*10)
        if args.benchmark == 'resnet':
            train_metadata, test_metadata = load_resnet_metadata(trial_id)
        elif args.benchmark == 'xgb':
            train_metadata, test_metadata = load_xgb_metadata(trial_id)
        else:
            train_metadata, test_metadata = create_metadata(args, trial_id)

        ts = GreatSurrogate(train_metadata, test_metadata)
        ts.set_lbd(True, init_lbd)
        ts.set_params(alpha, rate)
        ts.init = 1

        acquisition_func = EI(ts, par=0.01)
        smbo = SMBO(test_metadata, acquisition_func, ts, num_initial_design=args.num_init)

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

            adtm[index].append(smbo.get_relative_error())
            acc[index].append(smbo.get_best_accuracy())
            if hasattr(ts, 'get_weights') and args.debug:
                tmp_w = ts.get_weights()
                print('|%.3f ' * len(tmp_w) % tuple(tmp_w))

    # Print the metrics.
    for i in range(steps):
        metric = [np.mean(acc[i]), np.std(acc[i]), np.mean(adtm[i]), np.std(adtm[i])]
        print('{0}: acc mean|acc std|adtm mean|adtm std ==> {1}'.format('RTL-HPO', str(metric)))

    # Save all runtime metrics.
    runtime_metric_file = metric_format % (benchmark, benchmark, steps, args.num_rep,
                                           args.sample_ratio, alpha, init_lbd, rate)
    with open(runtime_metric_file, 'wb') as f:
        pickle.dump([acc, adtm], f)


if __name__ == "__main__":
    alpha = list(map(lambda x: float(x), args.alpha.split(',')))
    init_lbd = list(map(lambda x: float(x), args.init_lbd.split(',')))
    rate = list(map(lambda x: float(x), args.rate.split(',')))

    # Display hints about the hyperparameters.
    def hint():
        print('-'*50)
        print('Alpha Value   :', alpha)
        print('Initial Lambda:', init_lbd)
        print('Reduction Rate:', rate)
        print('-'*50)
    hint()

    configs = list()
    for t1 in alpha:
        for t2 in init_lbd:
            for t3 in rate:
                    configs.append((t1, t2, t3))
    # Parallelize configs in batch.
    # configs_num = len(configs)
    # for i in range(configs_num // batch_size + (1 if configs_num % batch_size != 0 else 0)):
    #     print('Start %d-th batch...' % i)
    #     run_parallel(configs[batch_size*i: batch_size*i + batch_size])
    run_parallel_async(eval_algo, configs, pool_size=args.batch)
    hint()
