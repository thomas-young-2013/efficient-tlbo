import itertools
import numpy as np
from scipy import stats
from sklearn import svm, linear_model, model_selection


def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(object):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """
    def __init__(self):
        self.clf = svm.SVC(kernel='linear', C=.1)
        self.coef = None

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)
        self.clf.fit(X_trans, y_trans)
        self.coef = self.clf.coef_.ravel() / np.linalg.norm(self.clf.coef_)

    def predict(self, X):
        if self.coef is not None:
            return np.dot(X, self.coef)
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        print(np.dot(X, self.coef).shape)
        print(y.shape)
        tau, _ = stats.kendalltau(np.dot(X, self.coef), y)
        return tau


if __name__ == '__main__':
    # as showcase, we will create some non-linear data
    # and print the performance of ranking vs linear regression

    np.random.seed(1)
    n_samples, n_features = 300, 5
    true_coef = np.random.randn(n_features)
    X = np.random.randn(n_samples, n_features)
    noise = np.random.randn(n_samples) / np.linalg.norm(true_coef)
    y = np.dot(X, true_coef)
    y = np.arctan(y)  # add non-linearities
    y += .1 * noise  # add noise
    Y = np.c_[y, np.mod(np.arange(n_samples), 5)]  # add query fake id

    kf = model_selection.KFold(n_splits=5, shuffle=True)
    # cv = model_selection.KFold(n_samples, 5)
    train, test = list(iter(kf))[-1]

    # make a simple plot out of it
    # import pylab as pl
    # pl.scatter(np.dot(X, true_coef), y)
    # pl.title('Data to be learned')
    # pl.xlabel('<X, coef>')
    # pl.ylabel('y')
    # pl.show()

    # print the performance of ranking
    rank_svm = RankSVM()
    rank_svm.fit(X[train], Y[train])
    print('Performance of ranking ', rank_svm.score(X[test], Y[test][:, 0]))
    # print(rank_svm.predict(X[test]))

    # and that of linear regression
    ridge = linear_model.RidgeCV(fit_intercept=True)
    ridge.fit(X[train], y[train])
    # X_test_trans, y_test_trans = transform_pairwise(X[test], y[test])
    # score = np.mean(np.sign(np.dot(X_test_trans, ridge.coef_)) == y_test_trans)
    score, _ = stats.kendalltau(np.dot(X[test], ridge.coef_), Y[test][:, 0])
    print('Performance of linear regression ', score)
