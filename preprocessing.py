import numpy as np
import pandas as pd
import torch as T

# TODO: make faulty data for testing
# TODO: generate more fault-free data - augmentation
# TODO: create a dataset object - mby faster


# 19 columns of data total
room_temp_columns = np.arange(9)  # columns of room temperature data
room_humidity_columns = np.arange(9, 17)  # columns of room humidity data
outside_temp_column = 17  # outside temperature
outside_humidity_column = 18  # outside humidity


def process_data(data):
    # load the data in predefined format from the csv file
    data = data.astype(str)  # to string format
    data = np.char.replace(data, ',', '.')  # change dashes to dots, really?
    data = data.astype(np.float64)  # convert to float
    return data


def f2c(data):
    # convert Fahrenheit to Celsius as
    # Tf = (Tc-32)*5/9
    return (data - 32) * 5 / 9


def load_data(fpath='DataCSV.csv'):
    # load data from the csv database
    raw_data = pd.read_csv(fpath, sep=';', dtype=str, header=0, index_col=0, skiprows=lambda x: x in [0,1,3,4], parse_dates=True)
    data_values = process_data(raw_data.to_numpy()[:, 1:20])  # relevant data converted to 2D float arrays
    data_labels = raw_data.columns[1:20]  # relevant data column labels
    data_timestamps = raw_data.index  # relevant data timestamps

    # transform temperature values from fahrenheit to celsius
    data_values[:, room_temp_columns] = f2c(data_values[:, room_temp_columns])
    data_values[:, outside_temp_column] = f2c(data_values[:, outside_temp_column])

    print('Data loaded succesfully')
    print(data_labels)
    print(data_values[1, :])

    return data_values, data_labels, data_timestamps

def prepare_for_training(data_values, train_size):
    # takes in data_values == columns of sensor sequences and spits out a large batch of training data with length
    # of sequences equal to train_size
    # data_values has a data sequence in each column
    data_values = np.array(data_values)
    n_X = data_values.shape[0] - train_size + 1  # total num of sub-sequences
    if data_values.ndim > 1:  # multiple columns
        trainX = np.array([data_values[i:i+train_size, :] for i in range(n_X)])
        trainX_T = T.from_numpy(trainX).float()
    else:
        trainX = np.array([data_values[i:i + train_size] for i in range(n_X)])
        trainX_T = T.from_numpy(trainX).float()
        trainX_T = T.unsqueeze(trainX_T, dim=-1)

    return trainX_T