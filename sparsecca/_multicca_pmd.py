import numpy as np
from scipy.linalg import svd

from ._utils_pmd import (
    binary_search,
    l2n,
    soft,
    scale,
    preprocess_datasets,
)


def get_crit(datasets, ws):
    crit = 0
    for ii in range(1, len(datasets)):
        for jj in range(0, ii):
            crit = crit + ws[ii].T @ datasets[ii].T @ datasets[jj] @ ws[jj]
    return crit


def update_w(datasets, idx, sumabs, ws, ws_final):
    tots = 0
    for jj in [ii for ii in range(len(datasets)) if ii != idx]:
        diagmat = ws_final[idx].T @ datasets[idx].T @ datasets[jj] @ ws_final[jj]
        for a in range(diagmat.shape[0]):
            for b in range(diagmat.shape[1]):
                if a != b:
                    diagmat[a, b] = 0

        tots = (
            tots
            + datasets[idx].T @ datasets[jj] @ ws[jj]
            - ws_final[idx] @ diagmat @ ws_final[jj].T @ ws[jj]
        )

    sumabs = binary_search(tots, sumabs)
    w_ = soft(tots, sumabs) / l2n(soft(tots, sumabs))

    return w_


def multicca(datasets, penalties, niter=25, K=1, standardize=True, mimic_R=True):
    """Re-implementation of the MultiCCA function from R package PMA.

    Params
    ------
    datasets : list
        List of n matrices of shape (samples x features)
    penalties : list
        List of n (1 x features) vectors. `c` in Witten 2009
    niter : int (default: 25)
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
    ws_final : list
        List of arrays of shape (datasets.shape[1], K) corresponding to the
        sparse canonical variates per dataset.
    ws_init : list(arr)
        List of arrrays of length `K` which contain the svd initializations for `w`.
    """
    datasets = datasets.copy()
    datasets = preprocess_datasets(datasets, standardize=standardize, mimic_R=mimic_R)

    ws = []
    for idx in range(len(datasets)):
        ws.append(svd(datasets[idx])[2][0:K].T)

    sumabs = []
    for idx, penalty in enumerate(penalties):
        if mimic_R:
            sumabs.append(penalty)
        else:
            sumabs.append(penalty * np.sqrt(datasets[idx].shape[1]))

    ws_init = ws

    ws_final = []
    for idx in range(len(datasets)):
        ws_final.append(np.zeros((datasets[idx].shape[1], K)))

    for comp_idx in range(K):
        ws = []
        for idx in range(len(ws_init)):
            ws.append(ws_init[idx][:, comp_idx])

        curiter = 0
        crit_old = -10
        crit = -20
        storecrits = []

        while (
            curiter < niter
            and np.abs(crit_old - crit) / np.abs(crit_old) > 0.001
            and crit_old != 0
        ):
            crit_old = crit
            crit = get_crit(datasets, ws)

            storecrits.append(crit)
            curiter += 1
            for idx in range(len(datasets)):
                ws[idx] = update_w(datasets, idx, sumabs[idx], ws, ws_final)

        for idx in range((len(datasets))):
            ws_final[idx][:, comp_idx] = ws[idx]

    return ws_final, ws_init
