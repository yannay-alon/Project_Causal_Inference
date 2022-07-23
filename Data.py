from sklearn.model_selection import KFold
import pandas as pd
from typing import Tuple


def read_data(folder_path: str, dataset_path: str, group: Tuple[int, ...]) -> Tuple[pd.DataFrame, float]:
    all_files = [rf"{folder_path}\{dataset_path}{num}.csv" for num in group]
    data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    ground_truth_suffix = "_cf"
    ate = pd.read_csv(rf"{folder_path}\{dataset_path}{group[0]}{ground_truth_suffix}.csv")["ATE"].mean()
    return data, ate


def split_train_test(data: pd.DataFrame, n_splits: int, shuffle: bool = True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle)

    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        yield train_data, test_data
