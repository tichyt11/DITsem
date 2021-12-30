import copy

import numpy as np
import pandas as pd
import torch as T

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


def create_batches(data_values, train_size):
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


def prepare_for_AE(data_values):
    X = data_values  # training data
    targets = T.unsqueeze(X[:, :, 0], 2)  # take first sensors as target
    return X, targets


def prepare_for_classifier(data_values):
    X = data_values  # training data
    p = X.size(0)//3  # partitions
    lin_X = add_linear_error(X[:p, :, :])
    outlier_X = add_outliers(X[p:2*p, :, :])
    offset_X = add_offset_error(X[2*p:, :, :])
    targets = T.cat((T.ones(X.size(0), 1), T.zeros(X.size(0), 1)), 0).float()
    X = T.cat((X, lin_X, outlier_X, offset_X), 0)
    return X, targets


def prepare_for_full_classifier(data_values):
    healthy = data_values  # training data
    lin_X = add_linear_error(healthy)
    off_X = add_offset_error(healthy)
    out_X = add_outliers(healthy)
    lin_off = add_linear_error(add_offset_error(healthy))
    lin_out = add_linear_error(add_outliers(healthy))
    off_out = add_outliers(add_offset_error(healthy))
    lin_off_out = add_linear_error(add_outliers(add_offset_error(healthy)))
    X = T.cat((healthy, lin_X, off_X, out_X, lin_off, lin_out, off_out, lin_off_out), 0)
    # create labels for all 8 'classes'
    n = data_values.size(0)
    healthy_l = T.zeros(n, 3)
    lin_l = T.tensor([1, 0, 0]).repeat(n, 1)
    off_l = T.tensor([0, 1, 0]).repeat(n, 1)
    out_l = T.tensor([0, 0, 1]).repeat(n, 1)
    lin_off_l = T.tensor([1, 1, 0]).repeat(n, 1)
    lin_out_l = T.tensor([1, 0, 1]).repeat(n, 1)
    off_out_l = T.tensor([0, 1, 1]).repeat(n, 1)
    lin_off_out_l = T.ones(n, 3)
    targets = T.cat((healthy_l, lin_l, off_l, out_l, lin_off_l, lin_out_l, off_out_l, lin_off_out_l), 0)
    return X, targets

def add_offset_error(data_values, minval=1, maxval=2):
# def add_offset_error(data_values, minval=2, maxval=3.5):  # orig
# def add_offset_error(data_values, minval=7, maxval=15):  # hum
    # take in fault-free data add offset error to first channel
    faulty_data = copy.deepcopy(data_values)  # create a copy
    offsets = minval + (maxval - minval)*T.rand(faulty_data.size(0), 1)
    signs = T.randint(0, 2, (data_values.size(0), 1))*2 - 1  # negative or positive slopes
    offsets = T.mul(signs, offsets)  # multiply element-wise
    faulty_data[:, :, 0] += offsets*T.ones(1, faulty_data.size(1))
    return faulty_data

def add_linear_error(data_values, minslope=0.06, maxslope=0.1):
# def add_linear_error(data_values, minslope=0.075, maxslope=0.15): # orig
# def add_linear_error(data_values, minslope=0.2, maxslope=0.4):  # hum
    # take in fault-free data and add a linear fcn
    faulty_data = copy.deepcopy(data_values)  # create a copy

    lin = T.arange(data_values.size(1))[None].float()  # 0,1,...,n_timesteps - 1
    slopes = minslope + (maxslope - minslope)*(T.rand(data_values.size(0), 1))  # slope magnitude for each batch
    signs = T.randint(0, 2, (data_values.size(0), 1))*2 - 1  # negative or positive slopes
    slopes = T.mul(signs, slopes)  # multiply element-wise
    lin_errors = T.matmul(slopes, lin)  # create linear functions for all batches

    faulty_data[:, :, 0] = faulty_data[:, :, 0] + lin_errors  # add linear errors to data
    return faulty_data

def add_outliers(data_values, minval=2.5, maxval=3.5):
# def add_outliers(data_values, minval=2.5, maxval=4):  # orig
# def add_outliers(data_values, minval=7, maxval=15):  # hum
    faulty_data = copy.deepcopy(data_values)  # create a copy

    indeces = T.randint(1, data_values.size(1) - 1, (data_values.size(0), 1)).flatten()  # random indeces
    signs = T.randint(0, 2, (data_values.size(0), 1)).flatten()*2 - 1  # negative or positive
    vals = minval + (maxval - minval)*T.rand(data_values.size(0), 1).flatten()
    vals = T.mul(signs, vals)  # multiply element-wise
    batches = T.arange(data_values.size(0))  # batch indeces
    faulty_data[batches, indeces, 0] += vals
    # faulty_data[batches, indeces + 1, 0] += vals
    # faulty_data[batches, indeces - 1, 0] += vals
    return faulty_data
