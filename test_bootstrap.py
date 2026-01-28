import numpy as np
import pandas as pd
import pytest

from BootsrapSE import BootstrapStandardError, get_bootstrap


 
#Inputs testing, Type, Value

def test_init_valid_inputs():
    bs = BootstrapStandardError(1, 2, 1, 100)
    assert bs.beta0 == 1
    assert bs.beta1 == 2
    assert bs.sigma == 1
    assert bs.N == 100

def test_init_invalid_beta0():
    with pytest.raises(TypeError):
        BootstrapStandardError("1", 2, 1, 100)

def test_init_invalid_beta1():
    with pytest.raises(TypeError):
        BootstrapStandardError(1, "2", 1, 100)

def test_init_invalid_sigma_negative():
    with pytest.raises(ValueError):
        BootstrapStandardError(1, 2, -1, 100)

def test_init_invalid_sigma_zero():
    with pytest.raises(ValueError):
        BootstrapStandardError(1, 2, 0, 100)

def test_init_invalid_sigma_type():
    with pytest.raises(TypeError):
        BootstrapStandardError(1, 2, "1", 100)

def test_init_invalid_N():
    with pytest.raises(ValueError):
        BootstrapStandardError(1, 2, 1, 1)

#_check_dataset

def test_check_dataset_not_dataframe():
    bs = BootstrapStandardError(1, 2, 1, 100)
    with pytest.raises(TypeError):
        bs._check_dataset([1, 2, 3])

def test_check_dataset_missing_columns():
    bs = BootstrapStandardError(1, 2, 1, 1000)
    D = pd.DataFrame({"x": [0,1,2,3,4,5]})
    with pytest.raises(ValueError):
        bs._check_dataset(D)

def test_check_dataset_valid():
    bs = BootstrapStandardError(1, 2, 1, 100)
    D = pd.DataFrame({
        "x": np.random.randn(10),
        "y": np.random.randn(10)
    })
    bs._check_dataset(D)

#_simulate_data
def test_simulate_data():
    bs = BootstrapStandardError(1, 2, 1, 100)
    data = bs._simulate_data()

    assert isinstance(data, pd.DataFrame)
    assert data.shape == (100, 2)
    assert "x" in data.columns and "y" in data.columns

def test_check_dataset_invalid(): 
    bs = BootstrapStandardError(1, 2, 1, 10) 
    with pytest.raises(TypeError): 
        bs._check_dataset([0.2,0.55]) 
    with pytest.raises(ValueError): 
        bs._check_dataset(pd.DataFrame({"a": [1], "b": [2]}))


def test_bootstrap_beta():
    bs = BootstrapStandardError(1, 2, 1, 1000)
    df = bs._simulate_data()

    beta1_hat, se_boot, beta1_boot = bs._bootstrap_beta(df, B=500)

    assert isinstance(beta1_hat, float)
    assert isinstance(se_boot, float)
    assert isinstance(beta1_boot, np.ndarray)
    assert len(beta1_boot) == 500

#_get_bootstrap

def test_bootstrap_beta_parallel():
    bs = BootstrapStandardError(1, 2, 1, 1000)
    df = bs._simulate_data()

    beta1_hat, se_boot, beta1_boot = bs._bootstrap_beta_parallel(df, B=500)

    assert isinstance(beta1_hat, float)
    assert isinstance(se_boot, float)
    assert isinstance(beta1_boot, np.ndarray)
    assert len(beta1_boot) == 500



def test_get_bootstrap():
    beta1_hat, se_boot, beta1_boot = get_bootstrap(
        beta0=1, beta1=2, sigma=1, N=100, B=30, useParallelism=False
    )

    assert isinstance(beta1_hat, float)
    assert isinstance(se_boot, float)
    assert isinstance(beta1_boot, np.ndarray)
    assert len(beta1_boot) == 30

def test_beta1_boot():
    np.random.seed(123)
    tolerance = 0.05

    bs = BootstrapStandardError(1, 2, 1, 1000)
    D = bs._simulate_data()

    beta1_hat, _, beta1_boot = bs._bootstrap_beta(D, B=(500))

    assert abs(np.mean(beta1_boot) - beta1_hat) < tolerance

