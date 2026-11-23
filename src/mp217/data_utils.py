"""Data utility module for MP217 Ration Shop Distribution project."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


def load_data(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame.

    Args:
        file_path: Path to the CSV dataset file.
        nrows: Optional number of rows to read. Useful for sampling.
        **kwargs: Additional keyword arguments passed to pandas.read_csv.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist at file_path.
        ValueError: If file_path is empty or points to a non-file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {path.resolve()}. "
            "Please verify the dataset path."
        )

    if not path.is_file():
        raise ValueError(f"Path does not point to a valid file: {path.resolve()}")

    try:
        df = pd.read_csv(path, nrows=nrows, **kwargs)
        return df
    except Exception as exc:
        raise IOError(f"Failed to read CSV dataset from {path.resolve()}: {exc}") from exc
