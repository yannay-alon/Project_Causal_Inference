import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Optional


def read_data(folder_path: str, dataset_path: str, group: Tuple[int, ...]) -> Tuple[pd.DataFrame, float]:
    all_files = [rf"{folder_path}\{dataset_path}{num}.csv" for num in group]
    data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    ground_truth_suffix = "_cf"
    ate = pd.read_csv(rf"{folder_path}\{dataset_path}{group[0]}{ground_truth_suffix}.csv")["ATE"].mean()
    return data, ate


def split_train_test(data: pd.DataFrame, target_name: str, num_splits: int, limit: Optional[int] = None):
    kf = StratifiedKFold(n_splits=num_splits)

    if limit is not None:
        data = data.head(limit)

    for train_index, test_index in kf.split(data, data[target_name]):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        yield train_data, test_data
