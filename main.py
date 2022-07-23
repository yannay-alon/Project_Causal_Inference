import numpy as np
from Data import read_data, split_train_test


def main():
    folder_path = "TestDatasets_lowD"
    dataset_path = "testdataset"
    groups = [(1, 5), (2, 6), (3, 7), (4, 8)]

    data, ate = read_data(folder_path, dataset_path, groups[0])

    num_splits = 5

    # All models to test
    models = {"model_name": ...}

    results = dict()

    for model_name, model in models.items():
        predicted_ate_values = []
        for train_data, test_data in split_train_test(data, num_splits):
            model.fit(train_data)

            predictions = model.predict(test_data)
            predicted_ate_values.append(model.calculate_ate(predictions))

        mean_ate = np.mean(predicted_ate_values)
        std_ate = np.std(predicted_ate_values)

        results[model_name] = {"mean": mean_ate, "std": std_ate}


if __name__ == '__main__':
    main()
