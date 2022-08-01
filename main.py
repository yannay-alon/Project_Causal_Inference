from matplotlib import pyplot as plt
from Data import read_data, split_train_test

from Models.IPW import *
from Models.DoublyRobust import *
from Models.XLearner import *
from Models.SLearner import *
from Models.TLearner import *
from Models.Matching import *


def main():
    folder_path = "TestDatasets_lowD"
    dataset_path = "testdataset"
    binary_groups = [(1, 5), (4, 8)]
    continuous_groups = [(2, 6), (3, 7)]

    data, ate = read_data(folder_path, dataset_path, binary_groups[1])
    num_features = 22
    treatment_name = "A"
    target_name = "Y"

    num_samples_values = [num for num in range(200, 1001, 100)]
    num_splits = 10

    # All models to test
    model_types = [IPW, BaselineIPW, XLearner, BaselineXLearner, DoublyRobust, SLearner, TLearner, Matching]
    models = {
        model.__name__: model(num_features, treatment_name, target_name) for model in model_types
    }

    for model_name, model in models.items():
        predicted_ate_means = []
        predicted_ate_stds = []

        for num_samples in num_samples_values:
            predicted_ate_values = []
            for train_data, test_data in split_train_test(data, num_splits, limit=num_samples):
                model.reset()
                model.fit(train_data)

                predicted_ate_values.append(model.calculate_ate(test_data))

            predicted_ate_means.append(np.mean(predicted_ate_values))
            predicted_ate_stds.append(np.std(predicted_ate_values) / 10)

        plt.errorbar(num_samples_values, predicted_ate_means, yerr=predicted_ate_stds, label=model_name)

    plt.plot(num_samples_values, [ate] * len(num_samples_values), label="True ATE", linestyle="dashed")

    plt.title("ATE v.s Num samples")
    plt.xlabel("Num samples")
    plt.ylabel("ATE")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
