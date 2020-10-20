import os
import argparse
import pickle as pkl
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import rankdata

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
parser.add_argument('--data_dir', type=str, default='./data/exp_results/online')
parser.add_argument('--transfer_trials', type=int, default=50)
parser.add_argument('--trial_num', type=int, default=50)
parser.add_argument('--rep_num', type=int, default=10)
args = parser.parse_args()

benchmark_id = args.algo_id
task_id = args.task_id
surrogate_type = args.surrogate_type
transfer_trials = args.transfer_trials
run_trials = args.trial_num
rep_num = args.rep_num
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
        if name.startswith('es') or name.startswith('obtl-idp_lc') or name == 'obtl':
            fill_values(name, 0)
        elif name.startswith('notl') or name.startswith('obtl-no_var'):
            fill_values(name, 1)
        elif name.startswith('rgpe') or name.startswith('obtl-gpoe'):
            fill_values(name, 2)
        elif name.startswith('rs') or name.startswith('obtlv-gpoe'):
            fill_values(name, 3)
        elif name.startswith('tst'):
            fill_values(name, 4)
        elif name.startswith('pogpe'):
            fill_values(name, 5)
        elif name.startswith('sgpr') or name.startswith('obtlv'):
            fill_values(name, 6)
        else:
            print(name)
            fill_values(name, 7)
    return color_dict, marker_dict


# def get_ranking(adtm_dict, num_ranking):
#     ranking_dict = {method: [] for method in adtm_dict.keys()}
#     for i in range(num_ranking):
#         _rank_dict = {}
#         value_dict = {}
#         for method in adtm_dict.keys():
#             value_dict[method] = adtm_dict[method][i]
#
#         sorted_item = sorted(value_dict.items(), key=lambda k: k[1])
#         cur_rank = 0
#         rank_gap = 1
#         for _idx, item in enumerate(sorted_item):
#             if cur_rank == 0:
#                 cur_rank += 1
#                 _rank_dict[item[0]] = cur_rank
#             else:
#                 if item[1] == sorted_item[_idx - 1][1]:
#                     _rank_dict[item[0]] = cur_rank
#                     rank_gap += 1
#                 else:
#                     cur_rank += rank_gap
#                     rank_gap = 1
#                     _rank_dict[item[0]] = cur_rank
#         counter = Counter(_rank_dict.values())
#         for method in adtm_dict.keys():
#             ranking_dict[method].append(
#                 (_rank_dict[method] * 2 + counter[_rank_dict[method]] - 1) / 2)
#     return ranking_dict

def get_ranking(adtm_dict, num_ranking):
    ranking_dict = {method: [] for method in adtm_dict.keys()}
    for i in range(num_ranking):
        _rank_dict = {}
        value_dict = {}
        for method in adtm_dict.keys():
            value_dict[method] = adtm_dict[method][i]

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
            ranking_dict[method].append(_rank_dict[method])
    return ranking_dict


if __name__ == "__main__":
    lw = 2
    ms = 6
    me = 1
    color_dict, marker_dict = fetch_color_marker(methods)

    adtm_dict = {}
    handles = list()
    ax = plt.subplot()
    try:
        rep_dict = {}
        for rep in range(rep_num):
            for idx, method in enumerate(methods):
                filename = '%s_%s_%d_%d_%s_%s_%d.pkl' % (method, benchmark_id, transfer_trials,
                                                         run_trials, surrogate_type, task_id, rep)
                if method.find('-') != -1:
                    _data_dir += 'fusion'
                else:
                    _data_dir = data_dir
                path = os.path.join(_data_dir, filename)
                with open(path, 'rb')as f:
                    array = pkl.load(f)

                adtm_array = [x[-1][0] for x in array[0]]

                adtm_dict[method] = adtm_array

                num_ranking = len(adtm_array)

            # print(adtm_dict)
            rank_dict = get_ranking(adtm_dict, num_ranking)

            for key in rank_dict:
                if key not in rep_dict:
                    rep_dict[key] = np.array(rank_dict[key])
                else:
                    rep_dict[key] = (np.array(rep_dict[key]) * rep + np.array(rank_dict[key])) / (rep + 1)

        _array = []
        for method in methods:
            _array.append(rep_dict[method])
        _array = np.array(_array)

        rank_array = []
        for id in range(_array.shape[1]):
            c = _array[:, id]
            rank_array.append(rankdata(c, method='min'))
        rank_array = np.array(rank_array)

        for id in range(rank_array.shape[1]):
            c = rank_array[:, id]
            print(Counter(c))
        exit()

        for idx, method in enumerate(methods):
            x = list(range(num_ranking))
            y = rep_dict[method]
            # print(x, y)
            label_name = r'\textbf{%s}' % (method.upper().replace('_', '-'))
            ax.plot(x, y, lw=lw,
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
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel('\\textbf{N-th Task} (%s)' % benchmark_id.replace('_', '\\_'), fontsize=15)

    ax.set_ylabel('\\textbf{Ranking}', fontsize=15)

    ax.set_xlim(0, len(x) + 1)
    plt.subplots_adjust(top=0.97, right=0.968, left=0.11, bottom=0.13)

    # plt.savefig('%s_%d_%d_%d_result.pdf' % (benchmark_id, runtime_limit, n_worker, rep_num))
    plt.show()
