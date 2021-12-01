import matplotlib.pyplot as plt
from random import randint
from models import LSTMAE, LSTMAE0
from preprocessing import load_data, room_temp_columns, prepare_for_training
from utils import load_model
import os
import torch as T


def plot_losses(losses):
    # plot epoch loss evolution
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss in epoch')
    plt.show()


def visualize(model, X, params):
    k = randint(0, X.size(0)-3)
    test_x = X[k:k+2, :, :]  # take a random sample from X
    with T.no_grad():
        test_y = model.eval().forward(test_x)  # push it through the model

    cmp = plt.cm.get_cmap('gist_rainbow', params['n_sensors'])

    for i in range(params['n_sensors']):
        plt.plot(range(params['n_timesteps']), test_x[-1, :, i].cpu().detach().numpy(), label='in %i' % (i+1), color=cmp(i))
        plt.scatter(range(params['n_timesteps']), test_y[-1, :, i].cpu().detach().numpy(), label='out %i' % (i+1), color=cmp(i))
    plt.legend()
    plt.show()


def visualize_folder(model, folder, X, params):
    # given a folder filled with params of the given model, loop through these and show reconstruction for each one
    k = randint(0, X.size(0)-3)
    test_x = X[k:k+2, :, :]  # take a random sample from X
    cmp = plt.cm.get_cmap('gist_rainbow', params['n_sensors'])

    fnames = os.listdir(folder)  # there must be only params in the folder
    for fname in fnames:
        model, checkpoint = load_model(model, os.path.join(folder, fname))
        with T.no_grad():
            test_y = model.eval().forward(test_x)  # push it through the model

        loss = checkpoint['loss']

        for i in range(params['n_sensors']):
            plt.plot(range(params['n_timesteps']), test_x[-1, :, i].cpu().detach().numpy(), label='in %i' % (i+1), color=cmp(i))
            plt.scatter(range(params['n_timesteps']), test_y[-1, :, i].cpu().detach().numpy(), label='out %i' % (i+1), color=cmp(i))
        plt.xlabel('t [h]')
        plt.title('prediction results of {} with loss {}'.format(fname, loss))
        plt.legend()
        plt.show()


if __name__=='__main__':
    params = {"epochs": 20000, "batchsize": 300, "lr": 0.007, "weight_decay": 0.0001, 'ID': 'ThirdTraining',
              'n_e_info': 100, 'n_sensors': 8, 'n_timesteps': 40}

    data_values, data_labels, data_timestamps = load_data('DataCSV.csv')
    room_temps = data_values[:, room_temp_columns]  # just the room temperatures in C

    X = prepare_for_training(room_temps[:, :params['n_sensors']], params['n_timesteps'])

    model = LSTMAE0(params['n_sensors'])
    target_folder = 'trained_models/LSTMAE0_40ts_8sens'
    visualize_folder(model, target_folder, X, params)

