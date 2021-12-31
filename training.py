from preprocessing import *
from visualization import *
from utils import *
import torch as T
from time import time
from datetime import datetime
import msvcrt

from models import *


def train(training_model, train_X, train_target, eval_X, eval_target, params, epochs, plot=True):
    b_size = params['batchsize']
    n_e_info = params['n_e_info']

    num_batches = train_X.size()[0]
    # num_iters = num_batches//b_size
    num_iters = num_batches//b_size + 1

    print('Starting training model {} at {}'.format(training_model.__class__.__name__, datetime.now().time()))
    t_start = time()
    epoch_losses = []
    eval_losses = []
    eval_percentages = []

    optimizer = T.optim.Adam(training_model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], eps=1e-4)
    criterion = params['criterion']

    for i in range(epochs):

        rand_indeces = T.randperm(train_X.size()[0])
        shuffled_X = train_X[rand_indeces]  # randomize the batches for each epoch
        shuffled_targets = train_target[rand_indeces]

        epoch_loss = 0
        for j in range(num_iters):
            batch_X = shuffled_X[j*b_size:(j+1)*b_size]
            batch_target = shuffled_targets[j*b_size:(j+1)*b_size]

            optimizer.zero_grad()
            batch_prediction = training_model.train().forward(batch_X)
            loss = criterion(batch_prediction, batch_target)
            loss.backward()
            optimizer.step()

            with T.no_grad():  # add loss to accumulated loss
                epoch_loss += loss.cpu()/(num_iters)  # current epoch loss per batch

        epoch_losses.append(epoch_loss)  # append to epoch losses list for later plotting

        if (i+1) % n_e_info == 0:  # print stats
            with T.no_grad():
                eval_prediction = training_model.eval().forward(eval_X)  # estimate of eval_X
                eval_loss = criterion(eval_target, eval_prediction)  # loss averaged over batches

                predicted_labels = (eval_prediction >= 0.5) * 1  # values >= 0.5 -> True
                correct_ids = T.sum(predicted_labels == eval_target, 1) == 3
                success_rate = 100*T.sum(correct_ids)/eval_prediction.size(0)  # total percentage of matches
            eval_percentages.append(success_rate)
            eval_losses.append(eval_loss)
            print("Epoch {}/{}, train loss: {:.4f}, eval loss: {:.4f}, eval success rate: {:.4f}%, finished after {} minutes".format(i+1, epochs,
                    epoch_loss, eval_loss, success_rate, int((time()-t_start)//60)))
            if (i+1) % (10*n_e_info) == 0:  # checkpoint every 10 stat infos
                save_model(training_model, eval_loss, params, i+1)

        if msvcrt.kbhit():  # handle quitting
            c = msvcrt.getch().decode('utf-8')
            if c == 'q':  # quit
                print('Canceling training process')
                break

    if plot:
       plot_losses(epoch_losses, eval_losses, n_e_info)

    return training_model


params = {"batchsize": 8192, "lr": 0.001, "weight_decay": 0.0001, 'epoch_0': 0,
            'n_e_info': 50, 'n_sensors': 8, 'n_timesteps': 20, 'target_folder': 'trained_models/FClassifier_1',
            'criterion': ''}

if __name__ == '__main__':

    data_values, data_labels, data_timestamps = load_data('DataCSV.csv')
    detect_sensor = 0

    sensor_ids = np.append(room_temp_columns, outside_temp_column)
    room_temps = data_values[:, sensor_ids]  # just the room temperatures in C
    plot_labels = np.array(data_labels[sensor_ids])
    room_temps[:, [0, detect_sensor]] = room_temps[:, [detect_sensor, 0]]  # switch places
    plot_labels[[0, detect_sensor]] = plot_labels[[detect_sensor, 0]]
    X = create_batches(room_temps[:, :params['n_sensors']], params['n_timesteps'])

    # room_hums = data_values[:, room_humidity_columns]  # just the room humidities
    # plot_labels = np.array(data_labels[room_humidity_columns])
    # room_hums[:, [0, detect_sensor]] = room_hums[:, [detect_sensor, 0]]  # switch places
    # plot_labels[[0, detect_sensor]] = plot_labels[[detect_sensor, 0]]
    # X = create_batches(room_hums[:, :params['n_sensors']], params['n_timesteps'])

    # train_X, train_targets = prepare_for_AE(X[:-2000, :, :].cuda())  # AE training data
    # eval_X, eval_targets = prepare_for_AE(X[-2000:-500, :, :].cuda())  # AE evaluation data
    # ver_X = X[-500:, :, :].cuda()  # AE verification data

    # prepare data for classifier training
    train_data = X[:-2000, :, :]
    eval_data = X[-2000:-500, :, :]
    ver_data = X[-500:, :, :]

    train_X, train_labels = prepare_for_full_classifier(train_data)
    eval_X, eval_labels = prepare_for_full_classifier(eval_data)
    ver_X, ver_labels = prepare_for_full_classifier(ver_data)

    idx = randint(0, train_data.size(0))
    print(idx)  # 11088
    plot_signals(train_data[0 + idx, :, :], plot_labels)
    plot_signals(train_X[7*train_data.size(0) + idx, :, :], plot_labels)

    model = FullClassifier_best(params['n_sensors']).float()  # load model
    print(model)
    # model, checkpoint = load_model(model, 'trained_models/FClassifier_1/Fclassifier_1_sen8_ts40_iter019000')
    # params['epoch_0'] = checkpoint['epoch']

    device = 'cuda'
    params['criterion'] = T.nn.MSELoss()
    params['lr'] = 0.001

    eval_full_classifier(model.eval().cpu(), train_X.cpu(), train_labels.cpu(), 1)
    model = train(model.to(device), train_X.to(device), train_labels.to(device), eval_X.to(device), eval_labels.to(device), params, epochs=10000)
    eval_full_classifier(model.eval().cpu(), train_X.cpu(), train_labels.cpu(), 0)
    eval_full_classifier(model.eval().cpu(), eval_X.cpu(), eval_labels.cpu(), 0)
    eval_full_classifier(model.eval().cpu(), ver_X.cpu(), ver_labels.cpu(), 0)

    params['criterion'] = T.nn.BCELoss()
    params['lr'] = 0.0001
    params['epoch_0'] = 10000

    model = train(model.to(device), train_X.to(device), train_labels.to(device), eval_X.to(device),
    eval_labels.to(device), params, epochs=5000)
    eval_full_classifier(model.eval().cpu(), train_X.cpu(), train_labels.cpu(), 0)
    eval_full_classifier(model.eval().cpu(), eval_X.cpu(), eval_labels.cpu(), 0)
    eval_full_classifier(model.eval().cpu(), ver_X.cpu(), ver_labels.cpu(), 0)

    params['criterion'] = T.nn.BCELoss()
    params['lr'] = 0.000001
    params['epoch_0'] = 15000

    model = train(model.to(device), train_X.to(device), train_labels.to(device), eval_X.to(device),
                  eval_labels.to(device), params, epochs=5000)
    eval_full_classifier(model.eval().cpu(), train_X.cpu(), train_labels.cpu(), 1)
    eval_full_classifier(model.eval().cpu(), eval_X.cpu(), eval_labels.cpu(), 1)
    eval_full_classifier(model.eval().cpu(), ver_X.cpu(), ver_labels.cpu(), 1)


# fault/healthy classifier training
    # eval_classifier(model, eval_data, T.full((eval_data.size(0), 1), 1.0))  # eval on outliers
    # zero_labels = T.full((eval_data.size(0), 1), 0.0)
    # eval_classifier(model, add_outliers(eval_data), zero_labels)  # eval on outliers
    # eval_classifier(model, add_linear_error(eval_data), zero_labels)  # eval on linear error
    # eval_classifier(model, add_offset_error(eval_data), zero_labels)  # eval on offset error
    # model = train(model.to(device), train_X.to(device), train_labels.to(device), eval_X.to(device), eval_labels.to(device), params)
    # eval_classifier(model.cpu(), train_X.cpu(), train_labels.cpu())
    # eval_classifier(model.cpu(), eval_X.cpu(), eval_labels.cpu())

# AE training
    # visualize_AE(model.cuda(), eval_data.cuda(), add_offset_error(eval_data.cpu()).cuda(), params)
    # visualize_AE(model.cuda(), eval_data.cuda(), add_linear_error(eval_data.cpu()).cuda(), params)
    # visualize_AE(model.cuda(), eval_data.cuda(), add_outliers(eval_data.cpu()).cuda(), params)
    # visualize_AE(model.cuda(), eval_data.cuda(), eval_X[eval_data.size(0):, :, :].cuda(), params)
    # params['criterion'] = T.nn.MSELoss()
    # model = train(model, train_X, train_targets, eval_X, eval_targets, params)
    # visualize_AE(model, train_X, add_linear_error(train_X.cpu()).cuda(), params)
    # visualize_AE(model, add_linear_error(eval_X.cpu()).cuda(), params)


