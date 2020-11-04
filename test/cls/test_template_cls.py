from functools import partial
import numpy as np
import sys
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

from cls_model import AdaboostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,\
    KNearestNeighborsClassifier, LDA, LightGBM, LibSVM_SVC, LibLinear_SVC, Logistic_Regression,\
    QDA, RandomForest

sys.path.append('../soln-ml')
from solnml.datasets.utils import load_data

import pickle
import argparse
from litebo.facade.bo_facade import BayesianOptimization as BO

model_map = dict(
    adaboost=AdaboostClassifier,
    extra=ExtraTreesClassifier,
    gb=GradientBoostingClassifier,
    knn=KNearestNeighborsClassifier,
    lda=LDA,
    lgb=LightGBM,
    libsvm=LibSVM_SVC,
    linear=LibLinear_SVC,
    logistic=Logistic_Regression,
    qda=QDA,
    rf=RandomForest,
)

model_name_map = dict(
    adaboost="adaboost",
    extra="extra_trees",
    gb="gradient_boosting",
    knn="k_nearest_neighbors",
    lda="lda",
    lgb="lightgbm",
    libsvm="libsvm_svc",
    linear="liblinear_svc",
    logistic="logistic_regression",
    qda="qda",
    rf="random_forest",
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=["adaboost", "extra", "gb", "knn", "lda", "lgb",
                                                  "libsvm", "linear", "logistic", "qda", "rf"])
parser.add_argument('--metric', type=str, default='bal_acc', choices=["bal_acc", "f1"])
parser.add_argument('--datasets', type=str)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--jobs', type=int, default=1)
parser.add_argument('--mode', type=str, default='bo')

args = parser.parse_args()
model_str = args.model
model_class = model_map[model_str]
metric = args.metric
dataset_str = args.datasets
run_count = args.n
jobs = args.jobs
mode = args.mode

task_type = 0   # cls


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset, '../soln-ml/', False, task_type=task_type)
        except Exception as e:
            print("Exception:", e)
            raise ValueError('Dataset - %s does not exist!' % _dataset)


def eval_func(params, x, y):
    params = params.get_dictionary()
    model = model_class(n_jobs=jobs, **params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == "bal_acc":
        return 1 - balanced_accuracy_score(y_test, y_pred)
    elif metric == 'f1':
        return 1 - f1_score(y_test, y_pred, average='macro')
    else:
        raise ValueError("unknown metric:", metric)


dataset_list = dataset_str.split(',')
check_datasets(dataset_list)
cs = model_class.get_cs()

_run_count = min(int(len(set(cs.sample_configuration(30000))) * 0.75), run_count)
print("run_count =", _run_count)
print("n_jobs =", jobs)

start_time = time.time()
for dataset in dataset_list:
    node = load_data(dataset, '../soln-ml/', True, task_type=task_type)
    _x, _y = node.data[0], node.data[1]
    eval = partial(eval_func, x=_x, y=_y)
    bo = BO(eval, cs, max_runs=_run_count, time_limit_per_trial=600, sample_strategy=mode, rng=np.random.RandomState(1))
    bo.run()

    save_model_name = model_name_map[model_str]
    dir_path = 'logs/%s_cls_%d/hpo_%s_%s/' % (mode, run_count, metric, save_model_name)
    file = '%s-%s-%s-hpo.pkl' % (dataset, save_model_name, metric)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, file), 'wb')as f:
        pickle.dump(bo.get_history().data, f)
    print("="*5, mode, run_count, metric, save_model_name, dataset, "cls finished", "="*5)

end_time = time.time()
m, s = divmod(end_time - start_time, 60)
h, m = divmod(m, 60)
print("Total time = %d hours, %d minutes, %d seconds." % (h, m, s))
