import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from test.cls.cls_model import RandomForest, LightGBM

seeds = [4465, 3822, 4531, 8459, 6295, 2854, 7820, 4050, 280, 6983,
         5497, 83, 9801, 8760, 5765, 6142, 4158, 9599, 1776, 1656]


model_class_dict = dict(
    random_forest=RandomForest,
    lightgbm=LightGBM,
)


def get_online_obj(algo_name, dataset, data_dir, n_jobs=1):
    model_class = model_class_dict[algo_name]

    # from test.basic.utils import load_data
    sys.path.append('../mindware')
    from mindware.datasets.utils import load_data

    x, y, _ = load_data(dataset, data_dir=data_dir, datanode_returned=False, preprocess=True, task_type=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

    def objective_function(config):
        config_dict = config.get_dictionary()
        model = model_class(**config_dict, n_jobs=n_jobs, random_state=47)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        perf = 1 - balanced_accuracy_score(y_test, y_pred)
        return perf

    return objective_function
