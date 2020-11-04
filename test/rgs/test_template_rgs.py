from functools import partial
import numpy as np
import sys
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from rgs_model import AdaboostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,\
    KNearestNeighborsRegressor, LassoRegressor, LightGBM, LibSVM_SVR, LibLinear_SVR,\
    RandomForest, RidgeRegressor

sys.path.append('../soln-ml')
from solnml.datasets.utils import load_data

import pickle
import argparse
from litebo.facade.bo_facade import BayesianOptimization as BO

model_map = dict(
    adaboost=AdaboostRegressor,
    extra=ExtraTreesRegressor,
    gb=GradientBoostingRegressor,
    knn=KNearestNeighborsRegressor,
    lasso=LassoRegressor,
    lgb=LightGBM,
    libsvm=LibSVM_SVR,
    linear=LibLinear_SVR,
    rf=RandomForest,
    ridge=RidgeRegressor,
)

model_name_map = dict(
    adaboost="adaboost",
    extra="extra_trees",
    gb="gradient_boosting",
    knn="k_nearest_neighbors",
    lasso="lasso_regression",
    lgb="lightgbm",
    libsvm="libsvm_svr",
    linear="liblinear_svr",
    rf="random_forest",
    ridge="ridge_regression",
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=["adaboost", "extra", "gb", "knn", "lasso",
                                                  "lgb", "libsvm", "linear", "rf", "ridge"])
parser.add_argument('--metric', type=str, default='mse', choices=["mse", "r2"])
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

task_type = 4   # rgs


def check_datasets(datasets):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset, '../soln-ml/', False, task_type=task_type)
        except Exception as e:
            print("Exception:", e)
            raise ValueError('Dataset - %s does not exist!' % _dataset)


def eval_func(params, x, y):
    params = params.get_dictionary()
    model = model_class(njobs=jobs, **params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if metric == "mse":
        return mean_squared_error(y_test, y_pred)   # minimize
    elif metric == "r2":
        return 1 - r2_score(y_test, y_pred)
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
    node = load_data(dataset, '../soln-ml/', True, task_type=4)
    _x, _y = node.data[0], node.data[1]
    eval = partial(eval_func, x=_x, y=_y)
    bo = BO(eval, cs, max_runs=_run_count, time_limit_per_trial=600, sample_strategy=mode, rng=np.random.RandomState(1))
    bo.run()

    save_model_name = model_name_map[model_str]
    dir_path = 'logs/%s_rgs_%d/hpo_%s_%s/' % (mode, run_count, metric, save_model_name)
    file = '%s-%s-%s-hpo.pkl' % (dataset, save_model_name, metric)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, file), 'wb')as f:
        pickle.dump(bo.get_history().data, f)
    print("="*5, mode, run_count, metric, save_model_name, dataset, "rgs finished", "="*5)

end_time = time.time()
m, s = divmod(end_time - start_time, 60)
h, m = divmod(m, 60)
print("Total time = %d hours, %d minutes, %d seconds." % (h, m, s))
