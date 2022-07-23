import pandas as pd
from typing import Tuple


def read_data(folder_path: str, dataset_path: str, group: Tuple[int, ...]) -> Tuple[pd.DataFrame, float]:
    all_files = [rf"{folder_path}\{dataset_path}{num}.csv" for num in group]
    data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    ate = pd.read_csv(rf"{folder_path}\{dataset_path}{group[0]}_cf.csv")["ATE"].mean()
    return data, ate