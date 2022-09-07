import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Data import read_data, split_train_test

from Models import *

np.seterr(all='raise')


def run_models(data: pd.DataFrame, ate: float,
               num_features: int, treatment_name: str, target_name: str):
    num_samples_values = [num for num in range(700, len(data) + 1, 200)]
    num_splits = 8
    num_repetitions = 10

    folder = "results"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # All models to test
    model_types = [
        # IPW,
        # XLearner,
        # DoublyRobust,
        SLearner,
        TLearner,
        # Matching
    ]

    baseline_types = [
        # BaselineIPW,
        # BaselineXLearner,
        # BaselineDoublyRobust,
        BaselineSLearner,
        BaselineTLearner,
        # BaselineMatching
    ]

    models = {
        model.__name__: model(num_features, treatment_name, target_name) for model in model_types
    }
    baseline_models = {
        model.__name__: model(num_features, treatment_name, target_name) for model in baseline_types
    }

    fig, axes = plt.subplots(2)

    ate_lines = []
    legends = []
    for (model_name, model), (baseline_name, baseline_model) in zip(models.items(), baseline_models.items()):
        print(f"Model: {model_name}")
        model_predicted_ate_means = []
        model_predicted_ate_stds = []

        model_bias_means = []
        model_bias_stds = []

        baseline_predicted_ate_means = []
        baseline_predicted_ate_stds = []

        baseline_bias_means = []
        baseline_bias_stds = []

        for num_samples in num_samples_values:
            print(f"\t#Samples: {num_samples}")
            model_predicted_ate_values = []
            model_bias_values = []
            baseline_predicted_ate_values = []
            baseline_bias_values = []
            for train_data, test_data in split_train_test(data, target_name, num_splits, num_repetitions,
                                                          limit=num_samples):
                model.reset()
                model.fit(train_data)
                model_predicted_ate = model.calculate_ate(test_data)
                model_predicted_ate_values.append(model_predicted_ate)
                model_bias_values.append(abs(model_predicted_ate - ate))

                baseline_model.reset()
                baseline_model.fit(train_data)
                baseline_predicted_ate = baseline_model.calculate_ate(test_data)
                baseline_predicted_ate_values.append(baseline_predicted_ate)
                baseline_bias_values.append(abs(baseline_predicted_ate - ate))

            model_predicted_ate_means.append(np.mean(model_predicted_ate_values))
            model_predicted_ate_stds.append(np.std(model_predicted_ate_values))

            model_bias_means.append(np.mean(model_bias_values))
            model_bias_stds.append(np.std(model_bias_values))

            baseline_predicted_ate_means.append(np.mean(baseline_predicted_ate_values))
            baseline_predicted_ate_stds.append(np.std(baseline_predicted_ate_values))

            baseline_bias_means.append(np.mean(baseline_bias_values))
            baseline_bias_stds.append(np.std(baseline_bias_values))

        np.save(f"{folder}/{model_name}_predicted_ate_means.npy", model_predicted_ate_means)
        np.save(f"{folder}/{model_name}_predicted_ate_stds.npy", model_predicted_ate_stds)

        np.save(f"{folder}/{model_name}_bias_means.npy", model_bias_means)
        np.save(f"{folder}/{model_name}_bias_stds.npy", model_bias_stds)

        np.save(f"{folder}/{baseline_name}_predicted_ate_means.npy", baseline_predicted_ate_means)
        np.save(f"{folder}/{baseline_name}_predicted_ate_stds.npy", baseline_predicted_ate_stds)

        np.save(f"{folder}/{baseline_name}_bias_means.npy", baseline_bias_means)
        np.save(f"{folder}/{baseline_name}_bias_stds.npy", baseline_bias_stds)

        ate_line = axes[0].errorbar(num_samples_values, model_predicted_ate_means, yerr=model_predicted_ate_stds)[0]
        axes[0].errorbar(num_samples_values, baseline_predicted_ate_means, yerr=baseline_predicted_ate_stds,
                         color=ate_line.get_color(), linestyle="dashed")
        ate_lines.append(ate_line)
        legends.append(model_name)

        axes[1].errorbar(num_samples_values, model_bias_means, yerr=model_bias_stds, color=ate_line.get_color())
        axes[1].errorbar(num_samples_values, baseline_bias_means, yerr=baseline_bias_stds,
                         color=ate_line.get_color(), linestyle="dashed")

    real_ate = np.full(len(num_samples_values), ate)
    np.save(f"{folder}/real_ate.npy", real_ate)

    ate_lines.append(plt.plot(num_samples_values, real_ate, linestyle="dotted")[0])
    legends.append("True ATE")

    axes[0].set_title("ATE v.s Num samples")
    axes[0].set_xlabel("Num samples")
    axes[0].set_ylabel("ATE")

    axes[1].set_title("Bias v.s Num samples")
    axes[1].set_xlabel("Num samples")
    axes[1].set_ylabel("Bias")
    plt.tight_layout()
    plt.show()

    legend_figure = plt.figure("Legend")
    legend_figure.legend(ate_lines, legends)
    legend_figure.show()


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
