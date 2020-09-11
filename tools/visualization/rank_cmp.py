import random
import pylab
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', choices=['svm', 'resnet', 'xgboost'], default='svm')
parser.add_argument('--use_tex', type=bool, default=False)
parser.add_argument('--frac', type=float, default=1)
args = parser.parse_args()

sns.set_style(style='whitegrid')
# %matplotlib inline

if args.use_tex:
    plt.rc('text', usetex=True)
    plt.rc('font', size=15.0, family='serif')
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    pylab.rcParams['figure.figsize'] = (8.0, 4.0)


color_list = ['mediumpurple', 'cadetblue', 'green', 'red', 'orange', 'lightpink']
algo_name = ['TST-R', 'RGPE', 'VIZIER', 'RTL', 'NONE-TL']
algo_num = 5
frac = args.frac
benchmark = args.benchmark


def load_data():
    metrics = list()
    for i in range(0, algo_num):
        data = np.load('./data/%s_stats_metric_%d_%.3f.npy' % (benchmark, i, frac))
        metrics.append(data)
    return metrics


def plot_rank_loss():
    metrics = load_data()
    trial_num = metrics[0].shape[0]
    fig, ax = plt.subplots(1)
    # color_dict = dict()
    # for i, item in enumerate(sorted(dict_data.keys())):
    #     color_dict[item] = color_list[i]
    for i in range(0, algo_num):
        ax.plot(np.linspace(1, trial_num, trial_num), metrics[i][:, 2], lw=2, label=algo_name[i], color=color_list[i])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(trial_num//5))
    ax.set_title('Benchmark: %s' % benchmark)
    ax.legend(loc='upper right')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Average Rank Loss')
    ax.set_ylim(1, 50)
    # ax.set_xlim(min_time, max_time)
    plt.savefig('./data/' + 'rank_loss_%s_%.3f.pdf' % (benchmark, frac))
    plt.show()


def plot_average_rank():
    metrics = load_data()
    trial_num = metrics[0].shape[0]
    rank_result = list()

    for i in range(trial_num):
        perf = list()
        for j in range(algo_num):
            perf.append(metrics[j][i][2])
        indexs = np.argsort(perf)
        rank_value = [len(indexs)]*len(indexs)
        for i, item in enumerate(indexs):
            rank_value[item] = i+1
        rank_result.append(rank_value)

    rank_result = np.array(rank_result)
    fig, ax = plt.subplots(1)
    # color_dict = dict()
    # for i, item in enumerate(sorted(dict_data.keys())):
    #     color_dict[item] = color_list[i]
    for i in range(0, algo_num):
        ax.plot(np.linspace(1, trial_num, trial_num), rank_result[:, i], lw=2, label=algo_name[i], color=color_list[i])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_title('Benchmark: %s' % benchmark)
    ax.legend(loc='upper right')
    ax.set_xlabel('Number of trials')
    ax.set_ylabel('Average Rank')
    ax.set_ylim(0, algo_num+1)
    # ax.set_xlim(min_time, max_time)
    plt.savefig('./data/' + 'average_rank_%s_%.3f.pdf' % (benchmark, frac))
    plt.show()


if __name__ == "__main__":
    # plot_average_rank()
    plot_rank_loss()
