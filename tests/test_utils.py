# tests/test_utils.py
# Unit tests for functions in utils.py using pytest.

import pytest
import pandas as pd
import os
import yaml
# Correct relative import if tests/ is a subdirectory of the project root
try:
    # Assumes utils.py is in the parent directory of the tests directory
    from .. import utils
except ImportError:
     # Fallback for different execution contexts or structures
     import sys
     # Add the project root directory (parent of tests/) to the path
     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
     import utils


# --- Fixtures ---
# Fixtures provide reusable setup for tests (e.g., creating temp files)

@pytest.fixture(scope="function") # 'function' scope runs fixture for each test function
def temp_dir(tmp_path):
    """Creates a temporary directory using pytest's built-in tmp_path fixture."""
    return tmp_path

@pytest.fixture
def dummy_good_csv(temp_dir):
    """Creates a valid dummy CSV file in the temp directory."""
    filepath = temp_dir / "good_data.csv"
    data = {'Pregnancies': [1, 2], 'Glucose': [100.0, 120.5], 'Outcome': [0, 1], 'Age': [25, 35]}
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return str(filepath) # Return path as string

@pytest.fixture
def dummy_bad_csv_missing_col(temp_dir):
    """Creates a dummy CSV missing a required column."""
    filepath = temp_dir / "missing_col_data.csv"
    data = {'Pregnancies': [1, 2], 'Outcome': [0, 1]} # Missing Glucose, Age
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return str(filepath)

@pytest.fixture
def dummy_bad_csv_negative_age(temp_dir):
    """Creates a dummy CSV with invalid negative age."""
    filepath = temp_dir / "negative_age_data.csv"
    data = {'Pregnancies': [1], 'Glucose': [100.0], 'Outcome': [0], 'Age': [-5]}
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return str(filepath)

@pytest.fixture
def dummy_empty_csv(temp_dir):
    """Creates an empty dummy CSV file (header only)."""
    filepath = temp_dir / "empty_data.csv"
    with open(filepath, 'w') as f:
        f.write("Pregnancies,Glucose,Outcome,Age\n") # Just header
    return str(filepath)

@pytest.fixture
def dummy_good_yaml(temp_dir):
    """Creates a valid dummy YAML config file."""
    filepath = temp_dir / "good_config.yaml"
    config_data = {
        'data': {'file_path': 'data.csv'},
        'model': {'type': 'XGBoost'}
    }
    with open(filepath, 'w') as f:
        yaml.dump(config_data, f)
    return str(filepath)

@pytest.fixture
def dummy_bad_yaml(temp_dir):
    """Creates an invalid (malformed) dummy YAML file."""
    filepath = temp_dir / "bad_config.yaml"
    # Incorrect YAML format (e.g., missing colon)
    with open(filepath, 'w') as f:
        f.write("data\n  file_path: data.csv\nmodel type: XGBoost")
    return str(filepath)


# --- Tests for load_data_util ---

def test_load_data_success(dummy_good_csv):
    """Test loading a valid CSV successfully."""
    required_cols = ['Pregnancies', 'Glucose', 'Outcome', 'Age']
    df = utils.load_data_util(dummy_good_csv, required_columns=required_cols)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == required_cols
    assert df.shape == (2, len(required_cols))

def test_load_data_no_required_cols_check(dummy_good_csv):
    """Test loading without specifying required columns."""
    df = utils.load_data_util(dummy_good_csv) # No required_columns passed
    assert isinstance(df, pd.DataFrame)
    assert "Outcome" in df.columns # Basic check that it loaded

def test_load_data_file_not_found():
    """Test loading a non-existent file."""
    with pytest.raises(FileNotFoundError):
        utils.load_data_util("non_existent_file.csv")

def test_load_data_missing_required_col(dummy_bad_csv_missing_col):
    """Test loading a CSV missing a specified required column."""
    required_cols = ['Pregnancies', 'Glucose', 'Outcome'] # Glucose is missing in file
    with pytest.raises(ValueError, match="Missing required columns.*'Glucose'"):
        utils.load_data_util(dummy_bad_csv_missing_col, required_columns=required_cols)

def test_load_data_empty_file(dummy_empty_csv):
     """Test loading an empty CSV file (header only)."""
     with pytest.raises(ValueError, match="Data file is empty"):
         utils.load_data_util(dummy_empty_csv) # pandas might raise EmptyDataError, caught as ValueError

def test_load_data_invalid_value(dummy_bad_csv_negative_age):
     """Test loading data with a value failing a specific check (negative Age)."""
     with pytest.raises(ValueError, match="Found negative values in 'Age' column."):
          utils.load_data_util(dummy_bad_csv_negative_age)


# --- Tests for load_config_util ---

def test_load_config_success(dummy_good_yaml):
    """Test loading a valid YAML config file."""
    config = utils.load_config_util(dummy_good_yaml)
    assert isinstance(config, dict)
    assert 'data' in config
    assert config['model']['type'] == 'XGBoost'

def test_load_config_file_not_found():
    """Test loading a non-existent config file."""
    with pytest.raises(FileNotFoundError):
        utils.load_config_util("non_existent_config.yaml")

def test_load_config_invalid_yaml(dummy_bad_yaml):
    """Test loading an improperly formatted YAML file."""
    with pytest.raises(ValueError, match="Error parsing YAML file"):
        utils.load_config_util(dummy_bad_yaml)

def test_load_config_not_dict(temp_dir):
    """Test loading a YAML file that doesn't parse to a dictionary."""
    filepath = temp_dir / "list_config.yaml"
    with open(filepath, 'w') as f:
        f.write("- item1\n- item2") # A list, not a dictionary
    with pytest.raises(ValueError, match="Config file did not parse as a dictionary"):
        utils.load_config_util(str(filepath))