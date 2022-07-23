from Data import read_data


def main():
    folder_path = "TestDatasets_lowD"
    dataset_path = "testdataset"
    groups = [(1, 5), (2, 6), (3, 7), (4, 8)]

    data, ate = read_data(folder_path, dataset_path, groups[0])


if __name__ == '__main__':
    main()
