import numpy as np
from tlbo.facade.base_surrogate import BaseSurrogate


class NNSMFO(BaseSurrogate):
    def __init__(self, train_metadata, test_metadata, cov_amp=2, kernel_type='Matern', base_model='gp', nearest_n=5):
        BaseSurrogate.__init__(self, train_metadata, test_metadata,
                               cov_amp=cov_amp, kernel_type=kernel_type, normalize_output=False)
        self.base_model_type = base_model
        self.next_config = None
        self.mapping_list = list()
        self.nearest_list = list(range(self.historical_task_num))
        self.nearest_num = nearest_n
        self.buffer_x = list()
        for i in range(self.historical_task_num):
            X = self.train_metadata[i][:, 1:]
            y = self.train_metadata[i][:, 0]
            indexs = np.argsort(y)
            mapping = dict()
            x_set = set()
            for rank, index in enumerate(indexs):
                key = str(X[index])
                x_set.add(key)
                if key not in mapping:
                    mapping[key] = rank
            self.mapping_list.append(mapping)
            self.buffer_x.append(x_set)

    def index_row(self, X, item):
        for index, x in enumerate(X):
            if (item == x).all():
                return index
        return -1

    def compute_rank(self, y, y_pred):
        disorder_num = 0
        num = len(y)
        if num < 2:
            return disorder_num
        for i in range(num):
            for j in range(num):
                if i != j and (y[i] > y[j]) ^ (y_pred[i] > y_pred[j]):
                    disorder_num += 1
        return disorder_num/(num*(num-1))

    def train(self, X: np.ndarray, y: np.array):
        self.update_incumbent(X, y)

        # Update the k nearest neighbors data sets.
        dist_list = list()
        for i in range(self.historical_task_num):
            X_s = self.train_metadata[i][:, 1:]
            y_s = self.train_metadata[i][:, 0]
            y_left = list()
            y_right = list()
            for id, item in enumerate(X):
                if str(item) in self.buffer_x[i]:
                    index = self.index_row(X_s, item)
                    if index != -1:
                        y_left.append(y_s[index])
                        y_right.append(y[id])
            # Update distance estimation.
            dist_list.append(self.compute_rank(y_left, y_right) if len(y_left) != 0 else 0.)
        self.nearest_list = np.argsort(dist_list)[:self.nearest_num]

    # TODO: need test.
    def predict(self, X: np.array):
        rank_list = list()
        for x in X:
            rank = 0
            for s in self.nearest_list:
                if str(x) in self.buffer_x[s]:
                    rank += self.mapping_list[s][str(x)]
            rank_list.append(rank)
        return np.array(rank_list)
