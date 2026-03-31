import pandas as pd
from src.utils.config import RAW_CSV_PATH


def load_raw_data(csv_path=RAW_CSV_PATH) -> pd.DataFrame:
    """
    Load raw airline sequence data from CSV.
    """
    df = pd.read_csv(csv_path)
    return df