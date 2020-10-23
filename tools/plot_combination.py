import os
import argparse
import traceback
import pickle as pkl
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns

# sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)

plt.rc('font', size=15.0, family='sans-serif')
plt.rcParams['font.sans-serif'] = "Tahoma"

plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'gray'
plt.rcParams["legend.fontsize"] = 15
label_fontsize = 20


parser = argparse.ArgumentParser()
parser.add_argument('--surrogate_type', type=str, default='rf')
parser.add_argument('--exp_id', type=str, default='exp1')
parser.add_argument('--task_id', type=str, default='main')
parser.add_argument('--algo_id', type=str, default='random_forest')
parser.add_argument('--methods', type=str, default='obtlv-gpoe,obtlv-idp_lc,obtlv-no_var')
parser.add_argument('--data_dir', type=str, default='./data/exp_results/')
parser.add_argument('--transfer_trials', type=int, default=50)
parser.add_argument('--trial_num', type=int, default=75)
parser.add_argument('--metric', type=str, default='rank')
args = parser.parse_args()

benchmark_id = args.algo_id
task_id = args.task_id
exp_id = args.exp_id
surrogate_type = args.surrogate_type
transfer_trials = args.transfer_trials
run_trials = args.trial_num
methods = args.methods.split(',')
data_dir = args.data_dir
metric = args.metric

data_dir = 'data/exp_results/fusion/'


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    names_dict = dict()
    method_names = ['gPoE', 'IDP-LC', 'NO-VAR']
    color_list = ['red', 'orchid', 'royalblue', 'brown', 'purple', 'orange', 'yellowgreen', 'navy', 'green', 'black']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x', '+', 'H']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]
        names_dict[name] = method_names[idx]

    for name in m_list:
        if name.find('-gpoe') != -1:
            fill_values(name, 0)
        elif name.find('-idp_lc') != -1:
            fill_values(name, 1)
        elif name.find('-no_var') != -1:
            fill_values(name, 2)
        else:
            raise ValueError('Unexpected method - %s.' % name)
    return color_dict, marker_dict, names_dict


def get_mean_ranking(adtm_dict, idx, num_ranking):
    ranking_dict = {method: [] for method in adtm_dict.keys()}
    for i in range(num_ranking):
        _rank_dict = {}
        value_dict = {}
        for method in adtm_dict.keys():
            value_dict[method] = adtm_dict[method][i][idx][0]
        # print(value_dict)
        sorted_item = sorted(value_dict.items(), key=lambda k: k[1])
        cur_rank = 0
        rank_gap = 1
        for _idx, item in enumerate(sorted_item):
            if cur_rank == 0:
                cur_rank += 1
                _rank_dict[item[0]] = cur_rank
            else:
                if item[1] == sorted_item[_idx - 1][1]:
                    _rank_dict[item[0]] = cur_rank
                    rank_gap += 1
                else:
                    cur_rank += rank_gap
                    rank_gap = 1
                    _rank_dict[item[0]] = cur_rank
        counter = Counter(_rank_dict.values())
        for method in adtm_dict.keys():
            ranking_dict[method].append(
                (_rank_dict[method] * 2 + counter[_rank_dict[method]] - 1) / 2)
    mean_dict = {method: np.mean(ranking_dict[method]) for method in ranking_dict.keys()}
    std_dict = {method: np.std(ranking_dict[method]) for method in ranking_dict.keys()}
    return mean_dict, std_dict


if __name__ == "__main__":
    lw = 2
    ms = 6
    me = 5
    color_dict, marker_dict, names_dict = fetch_color_marker(methods)
    fig, ax = plt.subplots()

    adtm_dict = {}
    num_ranking = np.inf
    handles = list()
    try:
        for idx, method in enumerate(methods):
            filename = '%s_%s_%d_%d_%s_%s.pkl' % (method, benchmark_id, transfer_trials,
                                                  run_trials, surrogate_type, task_id)
            path = os.path.join(data_dir, filename)
            with open(path, 'rb')as f:
                array = pkl.load(f)

            label_name = r'\textbf{%s}' % names_dict[method]
            x = list(range(len(array[1])))
            if metric == 'adtm':
                y = array[1][:, 1]
                print(array[0].shape)
                print(method, np.std(array[0], axis=0)[:, 1])
                # print(x, y)
                ax.plot(x, y, lw=lw,
                        label=label_name, color=color_dict[method],
                        marker=marker_dict[method], markersize=ms, markevery=me
                        )

                line = mlines.Line2D([], [], color=color_dict[method], marker=marker_dict[method],
                                     markersize=ms, label=label_name)
                handles.append(line)
            elif metric == 'rank':
                adtm_dict[method] = array[0]
                num_ranking = len(array[0]) if len(array[0]) < num_ranking else num_ranking
            else:
                raise ValueError('Invalid metric - %s.' % metric)

        if metric == 'rank':
            ranking_dict = {method: [] for method in adtm_dict.keys()}
            ranking_std_dict = {method: [] for method in adtm_dict.keys()}
            for idx in range(len(x)):
                mean_ranking_dict, std_ranking_dict = get_mean_ranking(adtm_dict, idx, num_ranking)
                for method in adtm_dict.keys():
                    ranking_dict[method].append(mean_ranking_dict[method])
                    ranking_std_dict[method].append(std_ranking_dict[method])
            for idx, method in enumerate(methods):
                label_name = r'\textbf{%s}' % names_dict[method]
                ax.plot(x, ranking_dict[method], lw=lw,
                        label=label_name, color=color_dict[method],
                        marker=marker_dict[method], markersize=ms, markevery=me
                        )
                print('=' * 20)
                print(method, ranking_std_dict[method])
                line = mlines.Line2D([], [], color=color_dict[method], marker=marker_dict[method],
                                     markersize=ms, label=label_name)
                handles.append(line)

    except Exception as e:
        traceback.print_exc()
        print(e)

    legend = ax.legend(handles=handles, loc=1, ncol=3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlabel('\\textbf{Number of Trials', fontsize=label_fontsize)
    if metric == 'adtm':
        ax.set_ylabel('\\textbf{ADTM}', fontsize=label_fontsize)
        if benchmark_id == 'lightgbm':
            plt.subplots_adjust(top=0.97, right=0.968, left=0.16, bottom=0.13)
        else:
            plt.subplots_adjust(top=0.97, right=0.968, left=0.15, bottom=0.13)
    elif metric == 'rank':
        ax.set_ylabel('\\textbf{Average Rank}', fontsize=label_fontsize)
        # ax.set_ylim(1, len(methods))
        plt.subplots_adjust(top=0.97, right=0.968, left=0.11, bottom=0.13)

    plt.savefig(data_dir + 'combination_%s_%s_%d_%s_result.pdf' % (exp_id, benchmark_id, run_trials, metric))
    plt.show()
