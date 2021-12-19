import matplotlib.pyplot as plt
from random import randint
from models import *
from preprocessing import load_data, room_temp_columns, create_batches, add_linear_error, add_outliers
from utils import load_model
import os
import torch as T
import numpy as np
from time import sleep


def plot_histograms_AE(AE, healthy_X, faulty_X):

    healthy_real = healthy_X[:, :, 0]  # healthy real data
    faulty_real = faulty_X[:, :, 0]  # faulty real data

    with T.no_grad():
        healthy_pred = AE.eval().forward(healthy_X)[:, :, 0]  # prediction from healthy data
        faulty_pred = AE.eval().forward(faulty_X)[:, :, 0]  # prediction from faulty data

    healthy_logSE = T.log(T.sum(T.square(healthy_pred - healthy_real), 1)).cpu().numpy()  # sum of squared differences
    faulty_logSE = T.log(T.sum(T.square(faulty_pred - faulty_real), 1)).cpu().numpy()

    plt.hist(np.concatenate((healthy_logSE[None], faulty_logSE[None]), 0).T, bins=100,
             label=['Healthy data prediction errors', 'Faulty data prediction errors'])
    plt.legend()
    plt.show()

    plt.hist(np.concatenate((healthy_logSE[None], faulty_logSE[None]), 0).T, bins=100, cumulative=True, density=True,
             label=['Healthy data prediction errors', 'Faulty data prediction errors'])
    plt.legend()
    plt.show()


def plot_losses(losses):
    # plot epoch loss evolution
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss in epoch')
    plt.show()


def eval_classifier(model, X, labels):
    with T.no_grad():
        prediction = model.eval().forward(X)
    positives_pred = prediction[labels == 1]
    true_positives = 100*T.sum(positives_pred >= 0.5)/positives_pred.size(0)
    false_negatives = 100*T.sum(positives_pred < 0.5)/positives_pred.size(0)

    negatives_pred = prediction[labels == 0]
    true_negatives = 100*T.sum(negatives_pred < 0.5)/negatives_pred.size(0)
    false_positives = 100*T.sum(negatives_pred >= 0.5)/negatives_pred.size(0)

    print('Confusion matrix: ')
    print('     PP    PN')
    print('P [{:.2f}, {:.2f}]\nN [{:.2f}, {:.2f}]'.format(true_positives, false_negatives, false_positives, true_negatives))

    errors = T.abs(prediction - labels)
    print('average error: {:.2f}'.format(T.mean(errors)))


def visualize_AE(model, X, faulty_X, params):
    if X.size(0) > 3:
        k = randint(0, X.size(0)-3)
        test_x = X[k:k + 2, :, :]  # take a random sample from X
        faulty_test_x = faulty_X[k:k + 2, :, :]
    else:
        test_x = X
        faulty_test_x = faulty_X

    with T.no_grad():
        test_y = model.eval().forward(test_x)  # push it through the model
        faulty_test_y = model.eval().forward(faulty_test_x)

    plt.plot(range(params['n_timesteps']), test_x[-1, :, 0].cpu().detach().numpy(), label='in', color='red')
    plt.scatter(range(params['n_timesteps']), test_y[-1, :, 0].cpu().detach().numpy(), label='out', color='red')
    plt.legend()
    plt.show()

    plt.plot(range(params['n_timesteps']), faulty_test_x[-1, :, 0].cpu().detach().numpy(), label='in_faulty', color='red')
    plt.scatter(range(params['n_timesteps']), faulty_test_y[-1, :, 0].cpu().detach().numpy(), label='out_faulty', color='red')
    plt.legend()
    plt.show()

    plot_histograms_AE(model, X, faulty_X)


def visualize_folder(model, folder, X, params, labels=None):
    # given a folder filled with params of the given model, loop through these and show reconstruction for each one
    k = randint(0, X.size(0)-3)
    test_x = X[k:k+2, :, :]  # take a random sample from X
    faulty_x = add_outliers(test_x)
    all_faulty_x = add_outliers(X)
    criterion = T.nn.MSELoss()

    if labels == None:
        labels = [('in %i' % (i+1), 'out %i' % (i+1)) for i in range(['n_sensors'])]

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    fnames = os.listdir(folder)  # there must be only params in the folder
    for fname in fnames:
        model, checkpoint = load_model(model, os.path.join(folder, fname))
        with T.no_grad():
            test_y = model.eval().forward(test_x)  # push it through the model
            loss = criterion(T.unsqueeze(test_x[:, :, 0], 2), test_y)

            faulty_y = model.eval().forward(faulty_x)
            faulty_loss = criterion(T.unsqueeze(faulty_x[:, :, 0], 2), faulty_y)

        ax1.clear()
        ax2.clear()
        ax3.clear()

        ax1.plot(range(params['n_timesteps']), test_x[-1, :, 0].cpu().detach().numpy(), label=labels[0][0], color='red')
        ax1.scatter(range(params['n_timesteps']), test_y[-1, :, 0].cpu().detach().numpy(), label=labels[0][1], color='red')
        ax1.set_title('Faulty - epoch: {} loss: {:.5f}'.format(checkpoint['epoch'], loss.numpy()))
        ax1.legend()

        ax2.plot(range(params['n_timesteps']), faulty_x[-1, :, 0].cpu().detach().numpy(), label=labels[0][0], color='red')
        ax2.scatter(range(params['n_timesteps']), faulty_y[-1, :, 0].cpu().detach().numpy(), label=labels[0][1],
                    color='red')
        ax2.set_title('Healthy - epoch: {} loss: {:.5f}'.format(checkpoint['epoch'], faulty_loss.numpy()))
        ax2.legend()

        healthy_real = X[:, :, 0]  # healthy real data
        faulty_real = all_faulty_x[:, :, 0]  # faulty real data
        with T.no_grad():
            healthy_pred = model.eval().forward(X)[:, :, 0]  # prediction from healthy data
            faulty_pred = model.eval().forward(all_faulty_x)[:, :, 0]  # prediction from faulty data
        healthy_logSE = T.log10(
            T.sum(T.square(healthy_pred - healthy_real), 1)).cpu().numpy()  # sum of squared differences
        faulty_logSE = T.log10(T.sum(T.square(faulty_pred - faulty_real), 1)).cpu().numpy()
        ax3.hist(np.concatenate((healthy_logSE[None], faulty_logSE[None]), 0).T, bins=100, density=True, cumulative=True,
                 label=['Healthy data errors', 'Faulty data errors'])
        ax3.legend()
        ax3.set_title('Histogram of errors')
        plt.setp(ax3, xlabel='log10 SSE')

        plt.show(block=False)
        plt.pause(0.05)


if __name__=='__main__':
    params = {"epochs": 20000, "batchsize": 300, "lr": 0.007, "weight_decay": 0.0001, 'ID': 'ThirdTraining',
              'n_e_info': 100, 'n_sensors': 8, 'n_timesteps': 40}

    data_values, data_labels, data_timestamps = load_data('DataCSV.csv')
    room_temps = data_values[:, room_temp_columns]  # just the room temperatures in C
    room_temp_labels = data_labels[room_temp_columns]
    room_temp_labels = [(room_temp_labels[i], 'LSTM output') for i in range(len(room_temp_labels))]

    X, Y = create_batches(room_temps[:, :params['n_sensors']], params['n_timesteps'])
    eval_X = X[-2000:-500, :, :]  # evaluation data

    model = LSTMAE_dual3(params['n_sensors'])
    target_folder = 'trained_models/LSTMAE_dual3'
    visualize_folder(model, target_folder, eval_X, params, room_temp_labels)

