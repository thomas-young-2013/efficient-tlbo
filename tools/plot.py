import os
import argparse
import pickle as pkl
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns

sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)

plt.rc('font', size=16.0, family='sans-serif')
plt.rcParams['font.sans-serif'] = "Tahoma"

# plt.rcParams['figure.figsize'] = (8.0, 4.5)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'gray'
plt.rcParams["legend.fontsize"] = 16

parser = argparse.ArgumentParser()
parser.add_argument('--surrogate_type', type=str, default='rf')
parser.add_argument('--task_id', type=str, default='main')
parser.add_argument('--algo_id', type=str, default='random_forest')
parser.add_argument('--methods', type=str, default='notl,rgpe,es,rs')
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--transfer_trials', type=int, default=50)
parser.add_argument('--trial_num', type=int, default=20)
args = parser.parse_args()

benchmark_id = args.algo_id
task_id = args.task_id
surrogate_type = args.surrogate_type
transfer_trials = args.transfer_trials
run_trials = args.trial_num
methods = args.methods.split(',')
data_dir = args.data_dir


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    color_list = ['red', 'royalblue', 'green', 'brown', 'purple', 'orange', 'yellowgreen', 'purple']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]

    for name in m_list:
        if name.startswith('es'):
            fill_values(name, 0)
        elif name.startswith('notl'):
            fill_values(name, 1)
        elif name.startswith('rgpe'):
            fill_values(name, 2)
        elif name.startswith('rs'):
            fill_values(name, 3)
        else:
            print(name)
            fill_values(name, 7)
    return color_dict, marker_dict


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
    ranking_dict = {method: np.mean(ranking_dict[method]) for method in ranking_dict.keys()}
    return ranking_dict


if __name__ == "__main__":
    lw = 2
    ms = 6
    me = 5
    plt.figure(figsize=(10, 4.5), dpi=100)
    fig = plt.figure(1)
    color_dict, marker_dict = fetch_color_marker(methods)

    for _id, plot_type in enumerate(['ranking', 'adtm']):
        adtm_dict = {}
        num_ranking = np.inf
        handles = list()
        ax = plt.subplot(int('12%d' % (_id + 1)))
        try:
            for idx, method in enumerate(methods):
                filename = '%s_%s_%d_%d_%s_%s.pkl' % (method, benchmark_id, transfer_trials,
                                                      run_trials, surrogate_type, task_id)
                path = os.path.join("%sdata/exp_results" % data_dir, filename)
                with open(path, 'rb')as f:
                    array = pkl.load(f)
                label_name = r'\textbf{%s}' % (method.upper().replace('_', '-'))
                x = list(range(len(array[1])))
                if plot_type == 'adtm':
                    y = array[1][:, 1]
                    print(x, y)
                    ax.plot(x, y, lw=lw,
                            label=label_name, color=color_dict[method],
                            marker=marker_dict[method], markersize=ms, markevery=me
                            )

                    line = mlines.Line2D([], [], color=color_dict[method], marker=marker_dict[method],
                                         markersize=ms, label=label_name)
                    handles.append(line)
                elif plot_type == 'ranking':
                    adtm_dict[method] = array[0]
                    num_ranking = len(array[0]) if len(array[0]) < num_ranking else num_ranking

            if plot_type == 'ranking':
                ranking_dict = {method: [] for method in adtm_dict.keys()}
                for idx in range(len(x)):
                    mean_ranking_dict = get_mean_ranking(adtm_dict, idx, num_ranking)
                    for method in adtm_dict.keys():
                        ranking_dict[method].append(mean_ranking_dict[method])
                for method in adtm_dict.keys():
                    label_name = r'\textbf{%s}' % (method.upper().replace('_', '-'))
                    ax.plot(x, ranking_dict[method], lw=lw,
                            label=label_name, color=color_dict[method],
                            marker=marker_dict[method], markersize=ms, markevery=me
                            )
                    line = mlines.Line2D([], [], color=color_dict[method], marker=marker_dict[method],
                                         markersize=ms, label=label_name)
                    handles.append(line)

        except Exception as e:
            print(e)

        legend = ax.legend(handles=handles, loc=1, ncol=2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.set_xlabel('\\textbf{Number of Trials} (%s)' % benchmark_id.replace('_', '\\_'), fontsize=15)
        if plot_type == 'adtm':
            ax.set_ylabel('\\textbf{ADTM}', fontsize=15)
            plt.subplots_adjust(top=0.97, right=0.968, left=0.11, bottom=0.13)
        elif plot_type == 'ranking':
            ax.set_ylabel('\\textbf{Average Rank}', fontsize=18)
            ax.set_ylim(1, len(methods))
            plt.subplots_adjust(top=0.97, right=0.968, left=0.11, bottom=0.13)

    # plt.savefig('%s_%d_%d_%d_result.pdf' % (benchmark_id, runtime_limit, n_worker, rep_num))
    plt.show()
