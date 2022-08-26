import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from Data import read_data, split_train_test

from Models import *

np.seterr(all='raise')


def run_models(data: pd.DataFrame, ate: float,
               num_features: int, treatment_name: str, target_name: str):
    num_samples_values = [num for num in range(300, len(data) + 1, 200)]
    num_samples_values = [100]
    num_splits = 5

    # All models to test
    model_types = [
        IPW, BaselineIPW,
        XLearner, BaselineXLearner,
        DoublyRobust, BaselineDoublyRobust,
        SLearner, BaselineSLearner,
        TLearner, BaselineTLearner,
        Matching, BaselineMatching
    ]

    models = {
        model.__name__: model(num_features, treatment_name, target_name) for model in model_types
    }
    for model_name, model in models.items():
        print(f"Model: {model_name}")
        predicted_ate_means = []
        predicted_ate_stds = []

        for num_samples in num_samples_values:
            print(f"\t#Samples: {num_samples}")
            predicted_ate_values = []
            for train_data, test_data in split_train_test(data, target_name, num_splits, limit=num_samples):
                model.reset()
                model.fit(train_data)

                predicted_ate_values.append(model.calculate_ate(test_data))

            predicted_ate_means.append(np.mean(predicted_ate_values))
            predicted_ate_stds.append(np.std(predicted_ate_values))
        np.save(f"results/{model_name}_predicted_ate_means.npy", predicted_ate_means)
        np.save(f"results/{model_name}_predicted_ate_stds.npy", predicted_ate_stds)
        plt.errorbar(num_samples_values, predicted_ate_means, yerr=predicted_ate_stds, label=model_name)

    plt.plot(num_samples_values, [ate] * len(num_samples_values), label="True ATE", linestyle="dashed")

    plt.title("ATE v.s Num samples")
    plt.xlabel("Num samples")
    plt.ylabel("ATE")
    plt.legend(bbox_to_anchor=(0, -0.6, 1, 1), loc="lower left", mode="expand", ncol=3)
    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(42)

    dimension = "high"

    if dimension == "low":
        folder_path = "TestDatasets_lowD"
        binary_groups = [(1, 5), (4, 8)]
        continuous_groups = [(2, 6), (3, 7)]
    else:
        folder_path = "TestDatasets_highD"
        binary_groups = [(1,), (2,), (5,), (6,)]
        continuous_groups = [(3,), (4,), (7,), (8,)]

    dataset_path = "testdataset"

    data, ate = read_data(folder_path, dataset_path, binary_groups[1])

    num_features = len(data.columns) - 2
    treatment_name = "A"
    target_name = "Y"

    run_models(data, ate, num_features, treatment_name, target_name)


if __name__ == '__main__':
    main()
