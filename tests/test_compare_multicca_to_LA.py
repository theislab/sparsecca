import numpy as np
import pandas as pd
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
#from tests.sparsecca._multicca_pmd import multicca_LA

from sparsecca.sparsecca._multicca_pmd import multicca_LA

def test_compare_multicca_to_Linear_approach():
    datasets = [
        pd.read_csv("/workspaces/sparsecca/tests/data/multicca1.csv", sep=",", index_col=0).values,
        pd.read_csv("/workspaces/sparsecca/tests/data/multicca3.csv", sep=",", index_col=0).values,
    ]
    
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    if not rpackages.isinstalled("PMA"):
        utils.install_packages("PMA", verbose=True)

    r_pma_ws = robjects.r(
        """
        library("PMA")

        cls <- c(lat = "numeric", lon = "numeric")
        data1 <- read.table("tests/data/multicca1.csv", sep = ",", header = TRUE)
        rownames(data1) <- data1$X
        data1 <- data1[, 2:ncol(data1)]

        data2 <- read.table("tests/data/multicca3.csv", sep = ",", header = TRUE)
        rownames(data2) <- data2$X
        data2 <- data2[, 2:ncol(data2)]

        datasets <- list(data1, data2)
        res <- MultiCCA(
            datasets,
            type = "standard",
            penalty = 1.5,
            ncomponents = 1,
            standardize = TRUE
        )

        res$ws
        """
    )

    ws_LA = multicca_LA(datasets, [1.5, 1.5],standardize=True, niter=25)

    print("\nR weigth:")
    print(r_pma_ws)

    print("\nLA weight:")
    print(ws_LA)

    k=0
    
    
    # TODO: install solver "ipopt" on dev container
    # TODO: compare output to r function
    #tmp = np.zeros((len(r_pma_ws)))
    #print(tmp)
    #tmp2 = np.zeros((len(ws_LA[0,k]),1))
    #print(tmp2)
    for i in range(len(r_pma_ws)):
        #print(np.asarray(ws_LA[i,k]).reshape(5,1))
        #for j in range(len(ws_LA[i,k])):
         #   print(ws_LA[i,k][j])
            #tmp2[j]=(ws_LA[i,k][j])
        # reshape ws_la -> k = 0
        #print(tmp2)
        #tmp[i] = tmp2
        #print(tmp)
        #print((np.array(ws_LA[i,0])))
        #print((np.array(r_pma_ws[i])))
        #print((np.array(r_pma_ws[i])).flatten())
        print(np.array(ws_LA[i,0]).reshape(5,1))
        print(np.array(r_pma_ws[i]))
        assert np.allclose(np.array(ws_LA[i,0]).reshape(5,1), (np.array(r_pma_ws[i])), rtol=1e-10)

if __name__ == "__main__":
    test_compare_multicca_to_Linear_approach()
    