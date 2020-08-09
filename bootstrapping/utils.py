import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from bootstrapping.settings import RESULTS_DIR


def dump_pickles(data: Any, subfolder: str, file_part: str) -> Path:
    dt = datetime.now().strftime('%Y%m%d%H%M')
    pickle_file_pattern = f'{{}}_{dt}.pickle'
    if not (RESULTS_DIR / subfolder).is_dir():
        (RESULTS_DIR / subfolder).mkdir()
    file_path = (RESULTS_DIR / subfolder / pickle_file_pattern.format(file_part))
    with file_path.open('wb') as handle:
        pickle.dump(data, handle)
    return file_path


def load_pickles(file_path: Path) -> Any:
    with file_path.open('rb') as handle:
        return pickle.load(handle)
