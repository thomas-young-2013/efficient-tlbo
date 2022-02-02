import os
import sys
import argparse
import traceback
import pickle as pkl
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns
from prettytable import PrettyTable

sys.path.insert(0, '.')
from utils import seeds

# sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)

plt.rc('font', size=15.0, family='sans-serif')
plt.rcParams['font.sans-serif'] = "Tahoma"

plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'gray'
plt.rcParams["legend.fontsize"] = 7    # 15
label_fontsize = 20

parser = argparse.ArgumentParser()
parser.add_argument('--surrogate_type', type=str, default='gp')
parser.add_argument('--exp_id', type=str, default='exp1')
parser.add_argument('--task_id', type=str, default='main')
parser.add_argument('--algo_id', type=str, default='random_forest')
parser.add_argument('--methods', type=str, default='rs,notl,scot,rgpe,tst,tstm,pogpe,obtlv,sgpr')
parser.add_argument('--data_dir', type=str, default='./data/exp_results/')
parser.add_argument('--transfer_trials', type=int, default=50)
parser.add_argument('--trial_num', type=int, default=50)
parser.add_argument('--task_set', type=str, default='class1', choices=['class1', 'class2', 'full'])
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
args = parser.parse_args()

benchmark_id = args.algo_id
task_id = args.task_id
exp_id = args.exp_id
surrogate_type = args.surrogate_type
transfer_trials = args.transfer_trials
run_trials = args.trial_num
methods = args.methods.split(',')
data_dir = args.data_dir
task_set = args.task_set
rep = args.rep
start_id = args.start_id

if exp_id == 'exp1':
    if benchmark_id == 'adaboost':
        data_dir = 'data/exp_results/main_random_3_20000/'
    else:
        data_dir = 'data/exp_results/main_random_4_20000/'
    # run_trials = 50
elif exp_id == 'exp2':
    data_dir = 'data/exp_results/main_random_5_20000/'
    run_trials = 75
elif exp_id == 'exp3':
    data_dir = 'data/exp_results/source_etc_random_5_20000/'
    methods = ['tst', 'pogpe', 'rgpe', 'obtlv']
    run_trials = 50
elif exp_id == 'exp4':
    data_dir = 'data/exp_results/combination/'
elif exp_id == 'exp5':
    data_dir = 'data/exp_results/warm_random_29_20000'
elif exp_id == 'exptest':
    data_dir = 'data/exp_results/main_random_class1_5_20000'
else:
    raise ValueError('Invalid exp id - %s.' % exp_id)

if task_set == 'class1':
    datasets = ['kc1', 'pollen', 'madelon', 'winequality_white', 'sick']
elif task_set == 'class2':
    datasets = ['kc1', 'pollen', 'madelon', 'winequality_white', 'sick', 'quake',
                   'hypothyroid(1)', 'musk', 'page-blocks(1)', 'page-blocks(2)',
                   'satimage', 'segment', 'waveform-5000(2)']
else:
    raise ValueError(task_set)


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    names_dict = dict()
    method_ids = ['obtl', 'space', 'notl', 'scot', 'sgpr', 'tst', 'tstm', 'pogpe', 'rgpe']
    method_names = ['TOPO', 'SPACE', 'I-GP', 'SCoT', 'SGPR', 'TST', 'TST-M', 'POGPE', 'RGPE']
    color_list = ['red', 'orchid', 'royalblue', 'brown', 'purple', 'orange', 'yellowgreen', 'navy', 'green', 'black']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x', '+', 'H']

    undefined_cnt = 0

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]
        if name not in method_ids:
            names_dict[name] = name.replace('_', '\\_')
        else:
            names_dict[name] = method_names[idx]

    for name in m_list:
        if exp_id == 'exp4':
            if name.find('-gpoe') != -1:
                fill_values(name, 0)
            elif name.find('-idp_lc') != -1:
                fill_values(name, 1)
            elif name.find('-no_var') != -1:
                fill_values(name, 2)
            else:
                raise ValueError('Unexpected method - %s.' % name)
        elif exp_id in ['exp1', 'exp2', 'exp5', 'exp3', 'exptest']:
            if name == 'space':
                fill_values(name, 1)
            elif name == 'notl':
                fill_values(name, 2)
            elif name == 'scot' or name == 'topo_v3':
                fill_values(name, 3)
            elif name == 'sgpr':
                fill_values(name, 4)
            elif name == 'tst':
                fill_values(name, 5)
            elif name == 'tstm':
                fill_values(name, 6)
            elif name == 'pogpe':
                fill_values(name, 7)
            elif name == 'rgpe':
                fill_values(name, 8)
            else:
                if name.startswith('es') or name.startswith('obtl'):
                    fill_values(name, 0)
                else:
                    fill_values(name, undefined_cnt)
                    undefined_cnt = (undefined_cnt + 1) % len(color_list)
                    # raise ValueError('Invalid method - %s.' % name)
        else:
            raise ValueError('Invalid exp id - %s.' % exp_id)
    return color_dict, marker_dict, names_dict, method_ids


def get_subplot_num(n):
    x = int(np.floor(np.sqrt(n)))
    y = int(np.ceil(n / x))
    return x, y


if __name__ == "__main__":
    lw = 2
    ms = 6
    me = 5
    color_dict, marker_dict, names_dict, method_ids = fetch_color_marker(methods)

    print(names_dict)
    method_list = list()
    _orders = list()
    for _method in methods:
        if _method in method_ids:
            _orders.append(method_ids.index(_method))
        else:
            _orders.append(0)
    print(methods)
    _methods = zip(methods, _orders)
    methods = [item[0] for item in sorted(_methods, key=lambda x: x[1])]
    print(methods)

    nx, ny = get_subplot_num(n=len(datasets))
    plt.figure(figsize=(4 * ny, 3 * nx))

    head = [' '] + methods
    table = PrettyTable(head)

    for data_idx, dataset in enumerate(datasets):
        ax = plt.subplot(nx, ny, data_idx + 1)
        # fig, ax = plt.subplots()
        handles = list()
        row = [dataset]

        for idx, method in enumerate(methods):
            all_array = []
            for rep_id in range(start_id, start_id + rep):
                seed = seeds[rep_id]
                filename = '%s_%s_%d_%d_%s_%s_%d.pkl' % (method, benchmark_id, transfer_trials,
                                                         run_trials, surrogate_type, task_id, seed)
                path = os.path.join(data_dir, filename)
                with open(path, 'rb')as f:
                    array = pkl.load(f)
                all_array.append(array)
            array = []
            for i in range(len(all_array[0])):
                data = [arr[i] for arr in all_array]
                array.append(np.mean(data, axis=0))  # mean over repeats

            label_name = r'\textbf{%s}' % names_dict[method]
            x = list(range(len(array[1])))
            y = array[0][data_idx][:, 0]
            print(array[0].shape)
            print(method, np.std(array[0], axis=0)[:, 1])
            lw = 2 if method in method_ids else 1
            # print(x, y)
            ax.plot(x, y, lw=lw,
                    label=label_name, color=color_dict[method],
                    marker=marker_dict[method], markersize=ms, markevery=me
                    )

            line = mlines.Line2D([], [], lw=lw, color=color_dict[method], marker=marker_dict[method],
                                 markersize=ms, label=label_name)
            handles.append(line)

            item = np.round(array[0][data_idx][:, 1][-1], 6)  # last val perf
            row.append(item)

        # if exp_id == 'exp3':
        #     legend = ax.legend(handles=handles, loc=1, ncol=2)
        # else:
        #     legend = ax.legend(handles=handles, loc=1, ncol=3)
        legend = ax.legend(handles=handles, loc=1, ncol=1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.set_xlabel('\\textbf{Number of Trials', fontsize=label_fontsize)

        ax.set_ylabel('\\textbf{ADTM}', fontsize=label_fontsize)

        plt.title('[%s]-%s' % (args.algo_id.replace('_', '\\_'), dataset.replace('_', '\\_'),))
        # plt.subplots_adjust(top=0.97, right=0.968, left=0.16, bottom=0.13)

        table.add_row(row)

    print(table)

    plt.tight_layout(pad=0.2)
    # plt.savefig(os.path.join(data_dir, '%s_%s_%d_%s_result_adtm.pdf' % (exp_id, benchmark_id, run_trials, metric)))
    plt.show()
