import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Any, Optional

from bootstrapping.settings import RESULTS_DIR


def dump_pickles(data: Any, subfolder: str, file_part: str) -> Optional[Path]:
    dt = datetime.now().strftime('%Y%m%d%H%M')
    pickle_file_pattern = f'{{}}_{dt}.pickle'
    if not (RESULTS_DIR / subfolder).is_dir():
        (RESULTS_DIR / subfolder).mkdir()
    file_path = (RESULTS_DIR / subfolder / pickle_file_pattern.format(file_part))
    try:
        with file_path.open('wb') as handle:
            pickle.dump(data, handle)
        return file_path
    except Exception as e:
        print(f'Could not save data {file_part} due to {str(e)}')
        return None


def dump_ndarray(data: np.ndarray, subfolder: str, file_part: str) -> Optional[Path]:
    dt = datetime.now().strftime('%Y%m%d%H%M')
    file_pattern = f'{{}}_{dt}.dat'
    if not (RESULTS_DIR / subfolder).is_dir():
        (RESULTS_DIR / subfolder).mkdir()
    file_path = (RESULTS_DIR / subfolder / file_pattern.format(file_part))
    try:
        np.save(file_path.open('wb'), data, allow_pickle=False)
        return file_path
    except Exception as e:
        print(f'Could not save data {file_part} due to {str(e)}')
        return None


def load_pickles(file_path: Path) -> Any:
    with file_path.open('rb') as handle:
        return pickle.load(handle)
