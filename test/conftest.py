# Defines common fixtures and makes them available to all tests

import os

import cupy as cp
import numpy as np
import pytest

CUR_DIR = os.path.abspath(os.path.dirname(__file__))

@pytest.fixture(scope="session")
def test_data_path():
    return os.path.join(CUR_DIR, "test_data")

# only load from disk once per session, and we use np.copy for the elements,
# to ensure data in this loaded file stays as originally loaded
@pytest.fixture(scope="session")
def data_file(test_data_path):
    in_file = os.path.join(test_data_path, "tomo_standard.npz")
    return np.load(in_file)


@pytest.fixture
def ensure_clean_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    yield None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

@pytest.fixture
def host_data(data_file):
    return np.copy(data_file["data"])

@pytest.fixture
def data(host_data, ensure_clean_memory):
    return cp.asarray(host_data)
