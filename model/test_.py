import numpy as np
import xarray as xr
import joblib

from . import SMAD, EMAD, BCMAD, TernaryMAD

def test_smad():
    arr = np.ones((5, 100, 100))

    dataarray = xr.DataArray(arr, dims=('time', 'y', 'x'), coords={'time': list(range(5))})
    dataset = xr.Dataset(data_vars={'b{}'.format(i) : ((i+1) + dataarray) for i in range(6)})

    model = SMAD()
    result = model.compute(dataset)

    assert isinstance(result, xr.Dataset)

def test_emad():
    arr = np.ones((5, 100, 100))

    dataarray = xr.DataArray(arr, dims=('time', 'y', 'x'), coords={'time': list(range(5))})
    dataset = xr.Dataset(data_vars={'b{}'.format(i) : ((i+1) + dataarray) for i in range(6)})

    model = EMAD()
    result = model.compute(dataset)

    assert isinstance(result, xr.Dataset)

def test_bcmad():
    arr = np.ones((5, 100, 100))

    dataarray = xr.DataArray(arr, dims=('time', 'y', 'x'), coords={'time': list(range(5))})
    dataset = xr.Dataset(data_vars={'b{}'.format(i) : ((i+1) + dataarray) for i in range(6)})

    model = BCMAD()
    result = model.compute(dataset)

    assert isinstance(result, xr.Dataset)

def test_ternary():
    arr = np.ones((5, 100, 100))

    dataarray = xr.DataArray(arr, dims=('time', 'y', 'x'), coords={'time': list(range(5))})
    dataset = xr.Dataset(data_vars={'b{}'.format(i) : ((i+1) + dataarray) for i in range(6)})

    model = TernaryMAD()
    result = model.compute(dataset)

    assert isinstance(result, xr.Dataset)
