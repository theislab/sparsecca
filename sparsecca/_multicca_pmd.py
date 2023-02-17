import numpy as np
from scipy.linalg import svd
import pyomo.environ as pyo
from collections import defaultdict

from ._utils_pmd import binary_search, l2n, soft, scale

import pandas as pd


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

        tots = (tots + datasets[idx].T @ datasets[jj] @ ws[jj] -
                ws_final[idx] @ diagmat @ ws_final[jj].T @ ws[jj])

        sumabs = binary_search(tots, sumabs)
        w_ = soft(tots, sumabs) / l2n(soft(tots, sumabs))

        return w_
    

# Linear Programming 

def ObjRule(model):
    """Objective Function (4.3 in witten 2009)"""
    features = len(model.PC.data())
    samples = len(model.samples.data())
    #TODO: array from  w_i_k (for all pcs)
    return sum(
                sum((np.asarray([[model.w_i_k_f[idx, k, f] for f in model.PC.data()] for k in model.K.data()])
               @ np.asarray(xi).reshape(samples,features).T 
               @ np.asarray(xj).reshape(samples,features)
               @ np.asarray([[model.w_i_k_f[jdx, k, f] for f in model.PC.data()] for k in model.K.data()]).T)[r,c] 
               for r in model.K.data() for c in model.K.data())
               for idx, xi in enumerate(model.X) for jdx, xj in enumerate(model.X) if idx<jdx )
        


def multicca(datasets, penalties, niter=25, K=1, standardize=True, mimic_R=True):
    """Re-implementation of the MultiCCA Using Linear programming.

    Params
    ------
    datasets
    penalties
    niter : int (default: 25) -> ignored
    K : int (default: 1)
    standardize : bool (default: True)
        Whether to center and scale each dataset before computing sparse
        canonical variates.
    mimic_R : bool (default: True)
        Whether to mimic the R implementation exactly. Note that this flag can
        significantly change the resulting values.

    Returns
    -------
    ws : list
        List of arrays of shape (datasets.shape[1], K) corresponding to the
        sparse canonical variates per dataset.
    """
    # get only values from datsets
    datasets = [datasets[0].iloc[:,1:7].values, datasets[1].iloc[:,1:6].values]

    # preprocessing:
    datasets = datasets.copy()
    for data in datasets:
        if data.shape[1] < 2:
            raise Exception('Need at least 2 features in each dataset')

    if standardize:
        for idx in range(len(datasets)):
            if mimic_R:
                datasets[idx] = scale(datasets[idx], center=True, scale=True)
            else:
                datasets[idx] = scale(datasets[idx], center=True, scale=False)




    # Linear Programming
    model = pyo.ConcreteModel()
    
    # sets: 
    model.Idx = pyo.Set(initialize=range(len(datasets)))
    model.samples = pyo.Set(initialize=range(len(datasets[0])))
    model.PC = pyo.Set(initialize=range(len(datasets[0][0])))
    model.K = pyo.Set(initialize=range(K))
    model.X = pyo.Set(initialize=datasets) 

    # params: ci i in [1:N]
    model.c = pyo.Param(model.Idx, initialize=penalties)

    # variables: weights
    model.w_i_k_f = pyo.Var(model.Idx,model.K, model.PC, bounds=(0, 1), initialize=0.5)
    
    # Objective
    model.Obj = pyo.Objective(rule=ObjRule, sense=pyo.maximize)

    # constraints: lasso model.constraint_lasso = pyo.ConstraintList()
    model.constraint_lasso = pyo.ConstraintList()
    for i in model.Idx:
        model.constraint_lasso.add(sum(model.w_i_k_f[i,k,f] for k in model.K.data() for f in model.PC.data())<= model.c[i])
              
    # constraints: (2-norm)^2 ||wi||22 <=1
    model.constraint_norm2 = pyo.ConstraintList()
    for i in model.Idx:
        model.constraint_norm2.add(sum(model.w_i_k_f[i,k,f] * model.w_i_k_f[i,k,f] for k in model.K.data() for f in model.PC.data()) <= 1)
    
    nonLinearOpt =pyo.SolverFactory('ipopt')
    instance_non_linear = model.create_instance()
    res = nonLinearOpt.solve(instance_non_linear)
    model.solutions.load_from(res)


    w = defaultdict(list)
    for i in model.Idx:
        for k in model.K.data():
            for f in model.PC.data():
                w[i, k].append(instance_non_linear.w_i_k_f[i,k,f].value) # maybe just i as index?
    return w
