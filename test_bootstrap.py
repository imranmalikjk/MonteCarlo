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
