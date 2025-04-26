import pandas as pd
import os
import yaml

def load_config_util(config_path="config.yaml"):
    """Loads configuration from a YAML file (Helper Function)."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Perform checks after loading
        if not isinstance(config, dict):
            # Directly raise the specific error we expect here
             raise ValueError("Config file did not parse as a dictionary.")

        return config

    except FileNotFoundError: # Keep specific FileNotFoundError handling if needed
         raise # Re-raise the original FileNotFoundError
    except yaml.YAMLError as e:
        # Raise specific error for YAML parsing issues
        raise ValueError(f"Error parsing YAML file: {e}") from e
    except ValueError: # Catch the specific ValueError raised above
        raise # Re-raise the specific ValueError
    except Exception as e:
        # Catch any other unexpected exceptions
        # Avoid catching the ValueError we handled specifically above
        # You might want to log this unexpected error before raising
        # print(f"Unexpected error type caught: {type(e)}")
        raise RuntimeError(f"Unexpected error loading config: {e}") from e


def load_data_util(filepath, required_columns=None):
    """Loads data from CSV and performs basic validation (Helper Function)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"Error reading CSV file {filepath}: {e}") from e

    if df.empty:
        raise ValueError(f"Data file is empty: {filepath}")

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {filepath}: {missing_cols}")

    # Example: Add a simple cleaning step to test
    if 'Age' in df.columns:
        if not (df['Age'] >= 0).all():
            raise ValueError("Found negative values in 'Age' column.")

    return df