import sys
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--num_rep', type=int, default=16)
parser.add_argument('--sample_ratio', type=float, default=1.)
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--benchmark', choices=['svm', 'resnet', 'weka', 'xgb'], default='resnet')
parser.add_argument('--norm', type=str, default='2')
parser.add_argument('--lbd', type=str, default='1.')
parser.add_argument('--loss', type=str, default='0')
parser.add_argument('--mu', type=str, default='1.')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--num_init', type=int, default=3)
args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/RTL')
if args.mode == 'server':
    sys.path.append('/home/liyang/codes/RTS')

from tlbo.acquisition_function.ei import EI
from tlbo.transfer_methods.benchmark_smbo import SMBO
from tlbo.facade.transfer_surrogate import TransferSurrogate
from tlbo.utils.metadata_extract import create_metadata
from tlbo.utils.load_resnet_metadata import load_resnet_metadata
from tlbo.utils.load_xgb_metadata import load_xgb_metadata
from tlbo.utils.parallel_runner import run_parallel_async

benchmark = args.benchmark
frac = args.sample_ratio
steps = args.steps
metric_format = './data/%s_res/TRANS-%s_step-%d_rep-%d_sr_%.3f_norm-%d_lbd-%.5f_lt-%d_mu-%.4f.npy'


def eval_algo(params):
    norm, lbd, loss_type, mu = params
    print('%s --> norm|lbd|loss_type|mu|id: %s' % (benchmark, str(params)))

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

        ts = TransferSurrogate(train_metadata, test_metadata, loss_type=loss_type, lbd=lbd, fusion=False)
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
    runtime_metric_file = metric_format % (benchmark, benchmark, steps, args.num_rep, frac,
                                           norm, lbd, loss_type, mu)
    with open(runtime_metric_file, 'wb') as f:
        pickle.dump([acc, adtm], f)


if __name__ == "__main__":
    norms = list(map(lambda x: int(x), args.norm.split(',')))
    lambdas = list(map(lambda x: float(x), args.lbd.split(',')))
    losses = list(map(lambda x: int(x), args.loss.split(',')))
    mus = list(map(lambda x: float(x), args.mu.split(',')))

    # Display hints about the hyperparameters.
    def hint():
        print('-'*50)
        print('Norm  :', norms)
        print('Lambda:', lambdas)
        print('Loss  :', losses)
        print('Mu    :', mus)
        print('-'*50)
    hint()

    configs = list()
    for norm in norms:
        for lbd in lambdas:
            for loss in losses:
                for mu in mus:
                    configs.append((norm, lbd, loss, mu))
    # Parallelize configs in batch.
    # configs_num = len(configs)
    # for i in range(configs_num // batch_size + (1 if configs_num % batch_size != 0 else 0)):
    #     print('Start %d-th batch...' % i)
    #     run_parallel(configs[batch_size*i: batch_size*i + batch_size])
    run_parallel_async(eval_algo, configs, pool_size=args.batch)
    hint()