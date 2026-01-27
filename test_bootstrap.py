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
    bs = BootstrapStandardError(1, 2, 1, 100)
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
    assert "x" in data.columns
    assert "y" in data.columns