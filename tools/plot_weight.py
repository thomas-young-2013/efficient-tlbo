import os
import argparse
import pickle as pkl
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
plt.rcParams["legend.fontsize"] = 15

parser = argparse.ArgumentParser()
parser.add_argument('--surrogate_type', type=str, default='rf')
parser.add_argument('--task_id', type=str, default='main')
parser.add_argument('--algo_id', type=str, default='random_forest')
parser.add_argument('--methods', type=str, default='notl,rgpe,es,rs')
parser.add_argument('--data_dir', type=str, default='./data/exp_results/main_random_29_20000/')
parser.add_argument('--transfer_trials', type=int, default=50)
parser.add_argument('--trial_num', type=int, default=50)
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
    names_dict = dict()
    method_ids = ['obtl', 'rs', 'notl', 'scot', 'sgpr', 'tst', 'tstm', 'pogpe', 'rgpe']
    method_names = ['TOPO', 'RS', 'I-GP', 'SCoT', 'SGPR', 'TST', 'TST-M', 'POGPE', 'RGPE']
    color_list = ['red', 'orchid', 'royalblue', 'brown', 'purple', 'orange', 'yellowgreen', 'navy', 'green', 'black']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x', '+', 'H']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]
        names_dict[name] = method_names[idx]

    for name in m_list:
        if name == 'rs':
            fill_values(name, 1)
        elif name == 'notl':
            fill_values(name, 2)
        elif name == 'scot':
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
                raise ValueError('Invalid method - %s.' % name)
    return color_dict, marker_dict, names_dict, method_ids


if __name__ == "__main__":
    lw = 2
    ms = 6
    me = 5
    color_dict, marker_dict, _, _ = fetch_color_marker(methods)

    adtm_dict = {}
    handles = list()
    ax = plt.subplot()
    try:
        for idx, method in enumerate(methods):
            if method != 'pogpe':
                filename = 'w_%s_%s_%d_%d_%s_%s.pkl' % (method, benchmark_id, transfer_trials,
                                                        run_trials, surrogate_type, task_id)
                if method.find('-') != -1:
                    _data_dir += 'fusion'
                else:
                    _data_dir = data_dir
                path = os.path.join(_data_dir, filename)
                with open(path, 'rb')as f:
                    array = pkl.load(f)

                if method == 'notl':
                    _method = 'i-gp'
                elif 'obtl' in method:
                    _method = 'topo'
                else:
                    _method = method
                label_name = r'\textbf{%s}' % (_method.upper().replace('_', '-'))

                if method == 'obtlv':
                    print(array)
                y = array[5]

            else:
                label_name = r'\textbf{%s}' % ('pogpe'.upper().replace('_', '-'))
                y = [0.5] * (run_trials - 3)

            x = list(range(0, run_trials - 3))

            # print(x, y)
            ax.plot(x, y, lw=lw,
                    label=label_name, color=color_dict[method],
                    marker=marker_dict[method], markersize=ms, markevery=me
                    )

            line = mlines.Line2D([], [], color=color_dict[method], marker=marker_dict[method],
                                 markersize=ms, label=label_name)
            handles.append(line)
    except Exception as e:
        print(e)

    legend = ax.legend(handles=handles, loc=2, ncol=2)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))

    ax.set_xlabel('\\textbf{Number of Trials}', fontsize=20)
    ax.set_ylabel('\\textbf{Target Weight}', fontsize=20)

    # ax.set_xlim(0, len(x) + 1)
    ax.set_ylim(0, 1)
    ax.grid(False)
    plt.subplots_adjust(top=0.97, right=0.968, left=0.11, bottom=0.13)
    plt.savefig('weight_rf.pdf')
    plt.show()
