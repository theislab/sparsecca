import numpy as np
from scipy.linalg import svd
import pyomo.environ as pyo
from collections import defaultdict
#from sparsecca import _utils_pmd

from ._utils_pmd import binary_search, l2n, soft, scale

import pandas as pd

   

# Linear Programming 

def scale(mtx, center=True, scale=True):
    """
    Reimplement scale function from R
    """
    if not center:
        raise NotImplementedError('Scaling without centering not implemented')

    centered = mtx - np.mean(mtx, axis=0)
    if not scale:
        return centered

    # to replicate the R implementation of scale, we apply Bessel's
    # correction when calculating the standard deviation in numpy
    scaled = centered / centered.std(axis=0, ddof=1)
    return scaled


def preprocess_datasets(datasets:list):
    # preprocess data
    datasets = datasets.copy()
    # 2 features needed
    for data in datasets:
        if len(data[0]) < 2:
            raise Exception('Need at least 2 features in each dataset')

        # standardize if set TRUE
    if standardize:
        for idx in range(len(datasets)):
            if mimic_R:
                datasets[idx] = scale(datasets[idx], center=True, scale=True)
            else:
                datasets[idx] = scale(datasets[idx], center=True, scale=False)

            datasets[idx] = datasets[idx].tolist()
            
    return datasets


def ObjRule(model):
    """Objective Function (4.3 in witten 2009)"""
    features = len(model.PC.data())
    samples = len(model.samples.data())
    return sum(
                (np.asarray([model.w_i_f[idx, f] for f in model.PC.data()])[np.newaxis]
               @ np.asarray(xi).reshape(samples,features).T 
               @ np.asarray(xj).reshape(samples,features)
               @ np.asarray([model.w_i_f[jdx, f] for f in model.PC.data()])[np.newaxis].T)[0,0] 
               for idx, xi in enumerate(model.X) for jdx, xj in enumerate(model.X) if idx<jdx )


def do_linear_approach(datasets, penalties):
""" solves 4.3 of witten 2009 with linear programming approach
    -------
    Parameters:  
        datasets: N matrices [samples x features]
        penalties: list of length N for each Xi
    
    -------
    Returns: 
        w: defaultdict(list)
        - for each matrix Xn in datasets (n in [1:n]): n-> weights_vector
        - each weights_vector: list of length f (featuresize)
        - f = len(datasets[0][0])
    """

    model = pyo.ConcreteModel()

    # sets 
    model.Idx = pyo.Set(initialize=range(len(datasets)))
    model.samples = pyo.Set(initialize=range(len(datasets[0])))
    model.PC = pyo.Set(initialize=range(len(datasets[0][0])))
    model.X = pyo.Set(initialize=datasets)

    # params: ci i in [1:K]
    model.c = pyo.Param(model.Idx, initialize=penalties)

    # var
    model.w_i_f = pyo.Var(model.Idx, model.PC, bounds=(0, 1), initialize=0.5)

    # obj
    model.Obj = pyo.Objective(rule=ObjRule, sense=pyo.maximize)
    
    # constraints: lasso 
    model.constraint_lasso = pyo.ConstraintList()
    for i in model.Idx:
        model.constraint_lasso.add(sum(model.w_i_f[i,f] for f in model.PC.data())<= model.c[i])
        
    # constraints: (2-norm)^2 ||wi||22 <=1
    model.constraint_norm2 = pyo.ConstraintList()
    for i in model.Idx:
        model.constraint_norm2.add(sum(model.w_i_f[i,f] * model.w_i_f[i,f] for f in model.PC.data()) <= 1)
        
    # solving
    nonLinearOpt = pyo.SolverFactory('ipopt')
    instance_non_linear = model.create_instance()
    res = nonLinearOpt.solve(instance_non_linear)
    model.solutions.load_from(res)
    
    instance_non_linear.display()
    
    from collections import defaultdict
    w = defaultdict(list)
    for i in model.Idx:
        for f in model.PC.data():
            w[i].append(instance_non_linear.w_i_f[i,f].value) 
            
    return w



def iterative_process_K(datasets:list, penalties:list,  K:int):
    """ calculates K weights [1xN]
    -------
    Parameters:  
        datasets: N matrices [samples x features]
        penalties: list of length N for each Xi
        K: Amount of MCPs
    
    -------
    Returns
        weights : list
        - list of length K
        - each entry is a default dict: n -> weights_vector (1 x feature)
            (n is the index of the matrix Xn. n in [1:N])
        - feature = len(datasets[0][0])
    """
    sample_size = len(datasets[0])
    feature_amount = len(datasets[0][0])
    
    datasets_next = preprocess_datasets(datasets)
    weights = []
    
    k = 0
    while k < K:
        w = do_linear_approach(datasets_next, penalties)
        datasets_current = datasets_next
    
        w_samples = {}
        for w_n in w:
            w_sample = np.repeat(w[w_n],sample_size, axis=0).reshape(sample_size,feature_amount)
            w_samples[w_n] = w_sample

        datasets_next = []
        count=0
        for X_i in datasets_current:
            X_i_next = X_i - w_samples[count]
            datasets_next.append(X_i_next.tolist())
            count+=1
            
        weights.append(w)      
            
        k += 1
        
    return weights


def multicca_LA(datasets, penalties, niter=25, K=1, standardize=True, mimic_R=True):
    """Re-implementation of the MultiCCA Using Linear programming.
    
    ignores mimic_R, standadize and niter
   
    Returns
    -------
    ws : list
        - list of length K
        - each entry is a default dict: n -> weights_vector (1 x feature)
            (n is the index of the matrix Xn. n in [1:N])
    """
    return iterative_process_K(datasets, penalties, K)
    
