import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pylab
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_id', type=str, default='abalone')
args = parser.parse_args()

dataset_id = args.dataset_id

sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)

plt.rc('font', size=16.0, family='Times New Roman')
plt.rcParams['font.sans-serif'] = "Tahoma"

# plt.rcParams['figure.figsize'] = (8.0, 4.5)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'gray'
plt.rcParams["legend.fontsize"] = 16

# plt.switch_backend('agg')
dir = './'

id = '%s_heat' % dataset_id
with open(dir + '%s-adaboost.pkl' % dataset_id, 'rb')as f:
    content = pkl.load(f)

perfs = [a[1] for a in content]
configs = [a[0] for a in content]

abbr1 = list(np.logspace(-2, np.log10(2), 20).round(4))
abbr2 = list(np.linspace(50, 487, 20).astype(int))

perf_matrix = np.zeros((20, 20))

for i, config in enumerate(configs):
    id_x = abbr1.index(config['learning_rate'].round(4))
    id_y = abbr2.index(config['n_estimators'])
    perf_matrix[id_x, id_y] = perfs[i]

print(perf_matrix)

fig, ax = plt.subplots()

if dataset_id == 'abalone':
    vmin, vmax = 0.56, 0.645
elif dataset_id == 'pollen':
    vmin, vmax = 0.48, 0.516
elif dataset_id == 'sick':
    vmin, vmax = 0.85, 0.94
elif dataset_id == 'space_ga':
    vmin, vmax = 0.72, 0.78
elif dataset_id == 'quake':
    vmin, vmax = 0.48, 0.51
elif dataset_id == 'kc1':
    vmin, vmax = 0.5, 0.6
elif dataset_id == 'spambase':
    vmin, vmax = 0.89, 0.93
elif dataset_id == 'hypothyroid(2)':
    vmin, vmax = 0.965, 0.99

ax = sns.heatmap(perf_matrix[:-1, :], ax=ax, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                 linewidths=.5, xticklabels=5, yticklabels=4)
ax.set_xticklabels(abbr1)
ax.set_yticklabels(abbr2)

ax.set_ylabel('\\textbf{n\_estimators}', fontsize=15)
ax.set_xlabel('\\textbf{learning rate}', fontsize=15)

plt.show()
# fig.savefig('./cmp.pdf', dpi=300)
