import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pylab
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_id', type=str)
parser.add_argument('--gap', type=int, default=50)

args = parser.parse_args()
dataset_id = args.dataset_id
gap_num = args.gap

sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)

plt.rc('font', size=15.0, family='Times New Roman')
plt.rcParams['font.sans-serif'] = "Tahoma"

# plt.rcParams['figure.figsize'] = (8.0, 4.5)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'gray'
plt.rcParams["legend.fontsize"] = 16

# plt.switch_backend('agg')
dir = './'

with open(dir + '%s-%d-adaboost.pkl' % (dataset_id, gap_num), 'rb')as f:
    content = pkl.load(f)

perfs = [a[1] for a in content]
configs = [a[0] for a in content]

abbr1 = list(np.logspace(-2, np.log10(2), gap_num).round(4))
abbr2 = list(np.linspace(50, 500, gap_num).astype(int))

perf_matrix = np.zeros((gap_num, gap_num))

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

label_abbr1, label_abbr2 = [], []
for i, abbr in enumerate(abbr1):
    if i == len(abbr1) - 1:
        label_abbr1.append(0.2)
    elif i == len(abbr1) - 2:
        label_abbr1.append('')
        label_abbr2.append(500)
    elif i % (gap_num / 5) == 0:
        label_abbr1.append(abbr1[i])
        label_abbr2.append(abbr2[i])
    else:
        label_abbr1.append('')
        label_abbr2.append('')

ax = sns.heatmap(perf_matrix[:-1, :], ax=ax, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                 cbar_kws={'shrink': 0.5},
                 linewidths=.5, xticklabels=label_abbr1, yticklabels=label_abbr2)

# ax.set_xticklabels(label_abbr1)
# ax.set_yticklabels(label_abbr2)

ax.set_ylabel('\\textbf{n\_estimators}', fontsize=18)
ax.set_xlabel('\\textbf{learning rate}', fontsize=18)

plt.subplots_adjust(top=0.97, right=0.968, left=0.11, bottom=0.13)

# plt.show()
fig.savefig('./%s.pdf' % dataset_id, dpi=300)
