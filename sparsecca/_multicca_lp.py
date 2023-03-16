import numpy as np
from scipy.linalg import svd
from collections import defaultdict
import pyomo.environ as pyo

from ._utils_pmd import scale, preprocess_datasets
from ._utils_pmd import soft


def _ObjRule(model):
    """Objective Function (4.3 in witten 2009)"""
    features = len(model.F.data())
    return sum(
        (
            np.asarray([model.w_i_f[s, f] for f in model.F.data()])[np.newaxis]
            @ np.asarray(model.Y[(s, t)]).reshape(features, features)
            @ np.asarray([model.w_i_f[t, f] for f in model.F.data()])[np.newaxis].T
        )[0, 0]
        for s, t in model.st
    )


def _update_w_lp(covs, penalties, ws_init, n_datasets, n_samples, n_features):
    """Solves 4.3 of witten 2009 with linear programming approach

    Params
    ------
    covs: set of covariance matrices, Y_st = X_s.T @ X_t for s,t in [1:n], s < t
    penalties: list of length N for each Xi
    ws_init: initial values for ws, usually provided by svd
    n_datasets: number of datasets
    n_samples: number of samples
    n_features: number of features

    Returns
    -------
        w: defaultdict(list)
            - for each matrix Xn in datasets (n in [1:n]): n-> weights_vector
            - each weights_vector: list of length f (featuresize)
            - f = len(datasets[0][0])
    """
    model = pyo.ConcreteModel()

    # sets
    model.N = pyo.Set(initialize=range(n_datasets))
    model.S = pyo.Set(initialize=range(n_samples))
    model.F = pyo.Set(initialize=range(n_features))
    model.st = pyo.Set(initialize=list(covs.keys()))

    def _init_Y(m, s, t):
        return covs[(s, t)]

    model.Y = pyo.Set(model.st, initialize=_init_Y)

    # params: ci i in [1:K]
    model.c = pyo.Param(model.N, initialize=penalties)

    # var
    model.w_i_f = pyo.Var(model.N, model.F, initialize=0.5)
    for n in range(len(ws_init)):  # datasets
        for f in range(len(ws_init[0])):  # features
            model.w_i_f[n, f].value = ws_init[n][f][0]

    model.d_i_f = pyo.Var(model.N, model.F, initialize=0, domain=pyo.Binary)

    # obj
    model.Obj = pyo.Objective(rule=_ObjRule, sense=pyo.maximize)

    # constraints: lasso
    model.constraint_lasso = pyo.ConstraintList()
    for i in model.N:
        model.constraint_lasso.add(sum(model.w_i_f[i, f] for f in model.F.data()) <= 1.0)

    # constraints: (2-norm)^2 ||wi||_2^2 <=1
    model.constraint_norm2 = pyo.ConstraintList()
    for i in model.N:
        model.constraint_norm2.add(sum(model.w_i_f[i, f] * model.w_i_f[i, f] for f in model.F.data()) <= 1)

    # solving
    nonLinearOpt = pyo.SolverFactory("ipopt")
    instance_non_linear = model.create_instance()
    res = nonLinearOpt.solve(instance_non_linear)
    model.solutions.load_from(res)

    np.zeros((n_datasets, n_features, 1))
    w = np.zeros((n_datasets, n_features, 1))
    for i in model.N:
        for f in model.F.data():
            w[i, f, 0] = instance_non_linear.w_i_f[i, f].value

    return w


def lp_pmd(datasets, penalties, K=1, standardize=True, mimic_R=True):
    """Calculates K weights [1xN]

    Params
    ------
    datasets : list[arr]
        List of n matrices of shape (samples x features)
    penalties : list[int]
        List of n (1 x features) vectors. `c` in Witten 2009
    K : int (default: 1)
        Number of latent factors to calculate.
    standardize : bool (default: True)
        Whether to center and scale each dataset before computing sparse
        canonical variates.
    mimic_R : bool (default: True)
        Whether to mimic the R implementation exactly. Note that this flag can
        significantly change the resulting values.

    Returns
    -------
    ws_final : list(arr)
        List of arrays of shape (datasets.shape[1], K) corresponding to the
        sparse canonical variates per dataset.
    ws_init : list(arr)
        List of arrrays of length `K` which contain the svd initializations for `w`.

    """
    n_datasets = len(datasets)
    n_samples = len(datasets[0])
    n_features = len(datasets[0][0])

    # preprocessing for pyomo
    datasets = preprocess_datasets(datasets, standardize=standardize, mimic_R=mimic_R)

    # calculate covariance matrices
    covs_next = {}
    for i, X_i in enumerate(datasets):
        for j, X_j in enumerate(datasets):
            if i < j:
                covs_next[(i, j)] = X_i.T @ X_j

    covs_next = {(s, t): cov.tolist() for (s, t), cov in covs_next.items()}

    weights = []
    k = 0
    ws_inits = []
    for k in range(K):
        # slightly different initialization - recalculate the svd per K
        ws_init = []
        for idx in range(len(datasets)):
            ws_init.append(svd(datasets[idx])[2][0:K].T)
        ws_inits.append(np.array(ws_init))
        w = _update_w_lp(covs_next, penalties, ws_init, n_datasets, n_samples, n_features)

        covs_current = covs_next
        covs_next = {}
        for (s, t), Y_st in covs_current.items():
            Y_st = np.array(Y_st)
            Y_st_next = Y_st - (w[s].T @ Y_st @ w[t]) * (w[s] @ w[t].T)
            covs_next[(s, t)] = Y_st_next.tolist()

        weights.append(w)

    w_final = np.zeros((len(datasets), n_features, K))
    for k, w_k in enumerate(weights):
        for n, w_value in enumerate(w_k):
            for f, w_feature in enumerate(w_value):
                w_final[n][f][k] = w_feature

    ws_init = [
        v.reshape(K, n_features).T
        for v in np.concatenate(  # could probably be simpler
            [x[:, :, 0].reshape(len(datasets), n_features) for x in ws_inits], axis=1
        )
    ]

    return list(w_final), ws_init
