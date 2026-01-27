import numpy as np
import pandas as pd

from typing import Tuple
from joblib import Parallel, delayed

class BootstrapStandardError:

    def __init__(self, beta0: float, beta1: float, sigma: float, N: int):
        """
        Args:
            beta0 (float): Intercept
            beta1 (float): Slope coeeficient
            sigma (float): Standard Deviation of the error term
            N (int): Number of Observations to draw

        """
        self._validate_params(beta0, beta1, sigma, N)
        self.beta0 = beta0
        self.beta1 = beta1
        self.sigma = sigma
        self.N = N

    @staticmethod
    def _validate_params(beta0, beta1, sigma, N):
        """Check the parameters given by the user
        """

        if not isinstance(beta0, (int, float)):
            raise TypeError('beta0 should be a numeric value')
        if not isinstance(beta1, (int, float)):
            raise TypeError('beta1 should be a numeric value')
        if not isinstance(sigma, (int, float)):
            raise TypeError("sigma must be numeric")
        elif sigma <= 0:
            raise ValueError("sigma must be strictly positive")
        if not isinstance(N, int):
            raise TypeError("N must be an integer")
        elif N <= 1:
            raise ValueError("N must be greater than 1")
        
    def _check_dataset(self, D:pd.DataFrame):
        """Check is the DataFrame is correctly formatted

        Args:
            D (pd.DataFrame): (xi, yi) from a linear model

        """
        
        if not isinstance(D, pd.DataFrame):
            raise TypeError('D should be a DataFrame')
        if 'x' not in D.columns or 'y' not in D.columns:
            raise ValueError("The DataFrame should contains 'x' and 'y'")
        
    def _simulate_data(self)->pd.DataFrame:
        """Draw N observations {(xi, yi)} from the linear model
        yi = β0 + β1xi + εi, εi ∼ N (0, sigma2).

        Returns:
            pd.DataFrame: Contains the simulated (x, y) N pairs 
        """
        x = np.random.normal(0, 1, self.N)
        eps = np.random.normal(0, self.sigma, self.N)
        y = self.beta0 + self.beta1 * x + eps

        return pd.DataFrame({"x": x, "y":y})
    
    @staticmethod
    def _estimate_beta(D: pd.DataFrame)->list:
        """The Ordinary Least Squares (OLS) estimator of β = (β0,β1)

        Args:
            D (pd.DataFrame): (xi, yi) from a linear model

        Returns:
            list: OLS Estimator
        """
        x = D['x'].values
        y = D['y'].values
        X = np.column_stack((np.ones(len(x)),x))
        beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y)
        return beta_hat
    
    @staticmethod
    def _compute_se(beta1_boot:np.ndarray)->float:
        """Compute the Bootstrap Standard Error

        Args:
            beta1_boot (np.ndarray): Bootstrap's Beta1

        Returns:
            float: Standard Error
        """
        B = len(beta1_boot)
        mean = np.mean(beta1_boot)
        se = np.sqrt(np.sum((beta1_boot - mean)**2) / (B - 1))
        return se
    
    def _bootstrap_beta(self, D:pd.DataFrame, B:int) -> Tuple[float, float, np.ndarray]:
        """Perform a nonparametric bootstrapping to compute the Monte Carlo standard error

        Args:
            D (pd.DataFrame): (xi, yi) from a linear model
            B (int): Number of Bootstrap to compute

        Returns:
            Tuple[float, float, np.ndarray]: estimated beta1, standard error bootstrap, beta1 bootstrap
        """

        self._check_dataset(D)

        beta1_hat = self._estimate_beta(D)[1]
        beta1_boot = np.zeros(B)

        for b in range(B):
            idx = np.random.choice(len(D),len(D),replace=True)
            D_b = D.iloc[idx, :]
            beta_b = self._estimate_beta(D_b)
            beta1_boot[b] = beta_b[1]
        
        se_boot = self._compute_se(beta1_boot)

        return beta1_hat, se_boot, beta1_boot
    
    def single_bootstrap(self,D):
        idx = np.random.choice(len(D),len(D),replace=True)
        D_b = D.iloc[idx, :]
        return self._estimate_beta(D_b)[1]
        
    def _bootstrap_beta_parallel(self, D:pd.DataFrame, B:int)->Tuple[float, float, np.ndarray]:
        """Beta Bootstrap using parallelism

        Args:
            D (pd.DataFrame): (xi, yi) from a linear model
            B (int): Number of Bootstrap to compute

        Returns:
            Tuple[float, float, np.ndarray]: estimated beta1, standard error bootstrap, beta1 bootstrap
        """

        self._check_dataset(D)

        beta1_hat = self._estimate_beta(D)[1]

        beta1_boot = Parallel(n_jobs=-1)(delayed(self.single_bootstrap)(D) for _ in range(B))

        beta1_boot = np.array(beta1_boot)
        se_boot = self._compute_se(beta1_boot)

        return beta1_hat, se_boot, beta1_boot

    
def get_bootstrap(beta0:int, beta1:int, sigma:int, N:int, B:int, useParallelism:bool=False):

    bs = BootstrapStandardError(beta0, beta1, sigma, N)

    data = bs._simulate_data()

    print("Dataset head:")
    print(data.head())

    if useParallelism: 
        beta1_hat, se_boot, beta1_boot = bs._bootstrap_beta_parallel(data, B)
    else:
        beta1_hat, se_boot, beta1_boot = bs._bootstrap_beta(data, B)

    return beta1_hat, se_boot, beta1_boot
