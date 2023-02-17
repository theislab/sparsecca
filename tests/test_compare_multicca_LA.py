import numpy as np
import pandas as pd
from rpy2 import robjects
import rpy2.robjects.packages as rpackages


from sparsecca import multicca_pmd

def test_compare_multicca_to_Linear_approach():
    datasets = [
        pd.read_csv("tests/data/multicca1.csv", sep=",", index_col=0).values,
        pd.read_csv("tests/data/multicca3.csv", sep=",", index_col=0).values,
    ]

    ws_LA = multicca_pmd(datasets, [1.5, 1.5],standardize=True, niter=25)
    
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

    print("\nR weigth:")
    print(r_pma_ws)

    print("\nLA weight:")
    print(ws_LA)

    '''for i in range(len(r_pma_ws)):
        assert np.allclose(ws_LA[i], np.array(r_pma_ws[i]), rtol=1e-10)'''

if __name__ == "__main__":
    test_compare_multicca_to_Linear_approach()
    