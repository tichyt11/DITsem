import matplotlib.pyplot as plt
from random import randint
from models import *
from preprocessing import *
from utils import load_model
import os
import torch as T
import numpy as np


def plot_signals(data, labels=None):
    # plots all the signals from a sample segment from data
    if data.dim() == 3:
        k = randint(0, data.size(0))
        X = data[k, :, :]  # take a random sample from data
    else:
        X = data
    cmp = plt.cm.get_cmap('gist_rainbow', X.size(1))

    if labels is None:
        labels = ['signal % i' % (i + 1) for i in range(X.size(1))]

    for i in range(X.size(1)):
        plt.plot(range(X.size(0)), X[:, i].cpu().numpy(), label=labels[i], color=cmp(i))
    fig = plt.gcf()
    fig.set_size_inches(8, 4, forward=True)
    fig.set_dpi(200)
    plt.xlabel('time [h]', fontsize=12)
    plt.ylabel('Temperature [°C]', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.show()



def plot_losses(losses):
    # plot epoch loss evolution
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss in epoch')
    plt.show()


def plot_losses(losses, eval_losses, ps_t):
    # plot epoch loss evolution + eval losses
    plt.plot(np.arange(len(losses)), losses, label='train loss')
    plt.plot(ps_t*(1+np.arange(len(eval_losses))), eval_losses, label='evaluation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average loss in epoch')
    plt.show()


def eval_full_classifier(model, X, labels, n_samples=1, plt_labels=None):
    # take a model and labeled data and print out the accuracies + plot some random prediction examples
    with T.no_grad():
        prediction = model.eval().forward(X)

    if plt_labels is None:
        plt_labels = ['signal %i' % (i + 1) for i in range(X.size(2))]

    for i in range(n_samples):
        k = randint(0, X.size(0))  # random index
        pred_k = prediction[k, :]
        label_k = labels[k, :]

        sample = X[k, :, :]
        cmp = plt.cm.get_cmap('gist_rainbow', sample.size(1))
        for i in range(sample.size(1)):
            plt.plot(range(sample.size(0)), sample[:, i].cpu().numpy(), label=plt_labels[i], color=cmp(i))
        fig = plt.gcf()
        fig.set_size_inches(8, 5, forward=True)
        fig.set_dpi(200)
        pred_frmt = [round(i, 2) for i in pred_k.tolist()]
        lbl_frmt = [round(i, 2) for i in label_k.tolist()]
        plt.title('Predicted probabilities: {} \nTarget label: {} \n'.format(pred_frmt, lbl_frmt), fontsize=17)
        plt.xlabel('time [h]', fontsize=12)
        plt.ylabel('Temperature [°C]', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.show()

    predicted_labels = (prediction >= 0.5)*1  # values >= 0.5 -> True
    same = predicted_labels == labels  # nx3 matrix of True/False

    corr_d = T.sum(same[:, 0]*1)
    corr_off = T.sum(same[:, 1]*1)
    corr_out = T.sum(same[:, 2]*1)
    n = X.size(0)
    print('d_acc:{:.4f} , off_acc:{:.4f} , out_acc:{:.4f} '.format(corr_d/n, corr_off/n,corr_out/n))

    correct_ids = T.sum(same, 1) == 3
    correct_predictions = predicted_labels[correct_ids]  # only labels, that were correctly predicted
    wrong_predictions = labels[~correct_ids]  # only labels, that were not correctly predicted
    uniq_lbls, counts = np.unique(wrong_predictions, return_counts=True, axis=0)
    print(uniq_lbls)
    print(counts)
    n_correct = correct_predictions.size(0)  # total number of matches

    print('Correctly predicted {:.2f} % of samples'.format(100*n_correct/X.size(0)))
    avg_abs_err = T.mean(T.abs(prediction - labels), 0)  # average 3-dimensional absolute error
    print('average error: [{:.2f}, {:.2f}, {:.2f}]'.format(avg_abs_err[0], avg_abs_err[1], avg_abs_err[2]))


def visualize_folder_classifier(model, folder, X, labels, params):
    # go through a folder with stored model parameters and print the prediction accuracy for each one
    fnames = os.listdir(folder)  # there must be only params in the folder
    for fname in fnames:
        model, checkpoint = load_model(model, os.path.join(folder, fname))
        if checkpoint['n_timesteps'] == params['n_timesteps']:
            with T.no_grad():
                prediction = model.eval().forward(X)

            predicted_labels = (prediction >= 0.5) * 1  # values >= 0.5 -> True
            same = predicted_labels == labels  # nx3 matrix of True/False
            correct_ids = T.sum(same, 1) == 3
            correct_predictions = predicted_labels[correct_ids]  # only labels, that were correctly predicted
            n_correct = correct_predictions.size(0)  # total number of matches
            print('model: {} training eval loss: {:.4f}, correct pred: {:.2f}'.format(fname, checkpoint['loss'], 100 * n_correct / X.size(0)))

            # pred_faults = T.sum(prediction >= 0.9, 1) > 0  # at least 1 error with 0.9 confidence
            # pred_normal = T.sum(prediction > 0.1, 1) == 0  # no errors with 0.9 confidence
            # false_alarms = (T.sum(labels[pred_faults], 1) == 0)*1
            # false_negatives = (T.sum(labels[pred_normal], 1) > 0)*1  # pred normal on errors
            # print('False alarms percentage: {:.4f}%'.format(100*T.sum(false_alarms)/T.sum(pred_faults*1)))
            # print('False negatives: {}, Predicted negatives: {}, FNr {:.4f}'.format(T.sum(false_negatives),
            # T.sum(pred_normal*1), 100*T.sum(false_negatives)/T.sum(pred_normal*1)))


if __name__=='__main__':
    params = {"epochs": 20000, "batchsize": 300, "lr": 0.007, "weight_decay": 0.0001, 'ID': 'ThirdTraining',
              'n_e_info': 100, 'n_sensors': 8, 'n_timesteps': 40}

    # load data
    data_values, data_labels, data_timestamps = load_data('DataCSV.csv')
    room_temps = data_values[:, room_temp_columns]  # just the room temperatures in C
    room_temp_labels = data_labels[room_temp_columns]
    room_temp_labels = ['Bedroom 1', 'Bedroom 2', 'Bedroom 3', 'Master Bedroom', 'Bathroom 1', 'Bathroom 2', 'Kitchen', 'Hall']

    # split into segments and add errors, create labels
    X = generate_segments(room_temps[:, :params['n_sensors']], params['n_timesteps'])
    train_data = X[:-2000, :, :]
    train_X, train_labels = prepare_for_full_classifier(train_data)
    eval_data = X[-2000:-1000:, :, :]
    eval_X, eval_labels = prepare_for_full_classifier(eval_data)
    ver_data = X[-1000:, :, :]
    ver_X, ver_labels = prepare_for_full_classifier(ver_data)

    # just plot some random signals for the paper
    idx = randint(0, train_data.size(0))
    print(idx)  # 5807 6173 7937 these are nice
    plot_signals(train_data[0 + 7937, :, :], room_temp_labels)
    plot_signals(train_X[7 * train_data.size(0) + 7937, :, :], room_temp_labels)

    model = FullClassifier_best(params['n_sensors'])  # load model
    print(model)
    target_folder = 'trained_models/Experiment2'
    visualize_folder_classifier(model, target_folder, eval_X, eval_labels, params)

    model, checkpoint = load_model(model, 'trained_models/Experiment2/FullClassifier_best_sen8_ts40_iter008000')
    eval_full_classifier(model.eval().cpu(), ver_X.cpu(), ver_labels.cpu(), 1, room_temp_labels[:params['n_sensors']])

