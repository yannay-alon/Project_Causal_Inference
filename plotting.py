import numpy as np
from matplotlib import pyplot as plt

from Models import *


def main():
    model_types = [
        IPW,
        XLearner,
        DoublyRobust,
        # SLearner,
        TLearner,
        Matching
    ]

    baseline_types = [
        BaselineIPW,
        BaselineXLearner,
        BaselineDoublyRobust,
        # BaselineSLearner,
        BaselineTLearner,
        BaselineMatching
    ]

    model_names = [model.__name__ for model in model_types]
    baseline_names = [model.__name__ for model in baseline_types]

    folder = "results"
    min_samples, max_samples = 700, 1900
    sample_step = 200
    num_samples_values = [num for num in range(min_samples, max_samples + 1, sample_step)]

    lines = []
    legends = []
    for model_name, baseline_name in zip(model_names, baseline_names):
        model_means = np.load(f"{folder}/{model_name}_predicted_ate_means.npy")[2:]
        model_stds = np.load(f"{folder}/{model_name}_predicted_ate_stds.npy")[2:]

        baseline_means = np.load(f"{folder}/{baseline_name}_predicted_ate_means.npy")[2:]
        baseline_stds = np.load(f"{folder}/{baseline_name}_predicted_ate_stds.npy")[2:]

        line = plt.errorbar(num_samples_values, model_means, yerr=model_stds)[0]
        plt.errorbar(num_samples_values, baseline_means, yerr=baseline_stds,
                     color=line.get_color(), linestyle="dashed")
        lines.append(line)
        legends.append(model_name)

    real_ate = np.load(f"{folder}/real_ate.npy")[2:]

    lines.append(plt.plot(num_samples_values, real_ate, linestyle="dotted")[0])
    legends.append("True ATE")

    plt.title("ATE v.s Num samples")
    plt.xlabel("Num samples")
    plt.ylabel("ATE")
    plt.tight_layout()
    plt.show()

    legend_figure = plt.figure("Legend")
    legend_figure.legend(lines, legends)
    legend_figure.show()

    for model_name, baseline_name in zip(model_names, baseline_names):
        model_means = np.load(f"{folder}/{model_name}_bias_means.npy")[2:]
        model_stds = np.load(f"{folder}/{model_name}_bias_stds.npy")[2:]

        baseline_means = np.load(f"{folder}/{baseline_name}_bias_means.npy")[2:]
        baseline_stds = np.load(f"{folder}/{baseline_name}_bias_stds.npy")[2:]

        line = plt.errorbar(num_samples_values, model_means, yerr=model_stds)[0]
        plt.errorbar(num_samples_values, baseline_means, yerr=baseline_stds,
                     color=line.get_color(), linestyle="dashed")
        lines.append(line)
        legends.append(model_name)

    plt.title("Bias v.s Num samples")
    plt.xlabel("Num samples")
    plt.ylabel("Bias")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
