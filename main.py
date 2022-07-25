import numpy as np
from matplotlib import pyplot as plt
from Data import read_data, split_train_test

from IPW import *


def main():
    folder_path = "TestDatasets_lowD"
    dataset_path = "testdataset"
    groups = [(1, 5), (2, 6), (3, 7), (4, 8)]

    data, ate = read_data(folder_path, dataset_path, groups[0])
    num_features = 22
    treatment_name = "A"
    target_name = "Y"

    num_samples_values = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    num_splits = 5

    # All models to test
    models = {
        "IPW": IPW(num_features, treatment_name, target_name),
        "Baseline IPW": BaselineIPW(num_features, treatment_name, target_name)
    }

    for model_name, model in models.items():
        predicted_ate_means = []
        predicted_ate_stds = []

        for num_samples in num_samples_values:
            predicted_ate_values = []
            for train_data, test_data in split_train_test(data, num_splits, limit=num_samples):
                model.reset()
                model.fit(train_data)

                predictions = model.predict(test_data)
                predicted_ate_values.append(model.calculate_ate(test_data, predictions))

            predicted_ate_means.append(np.mean(predicted_ate_values))
            predicted_ate_stds.append(np.std(predicted_ate_values))

        plt.errorbar(num_samples_values, predicted_ate_means, yerr=predicted_ate_stds, label=model_name)

    plt.plot(num_samples_values, [ate] * len(num_samples_values), label="True ATE", linestyle="dashed")

    plt.title("ATE v.s Num samples")
    plt.xlabel("Num samples")
    plt.ylabel("ATE")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
