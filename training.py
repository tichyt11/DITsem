from preprocessing import load_data, room_temp_columns, outside_temp_column, create_batches, add_linear_error, add_outliers, prepare_for_AE
from models import LSTMAE_new2, LSTMAE_new3, LSTMAE_new4, LSTMAE_new5, LSTMAE_dual, LSTMAE_dual2, LSTMAE_dual3, Classifier
from visualization import visualize_AE, visualize_folder, plot_losses, plot_histograms_AE, eval_classifier
from utils import load_model, save_model
import torch as T
from time import time
from datetime import datetime
import msvcrt


def train(training_model, train_X, train_target, eval_X, eval_target, params, plot=True):
    b_size = params['batchsize']
    epochs = params['epochs']
    n_e_info = params['n_e_info']

    num_batches = train_X.size()[0]
    num_iters = num_batches//b_size

    print('Starting training model {} at {}'.format(training_model.__class__.__name__, datetime.now().time()))
    t_start = time()
    epoch_losses = []

    optimizer = T.optim.Adam(training_model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], eps=1e-4)
    criterion = params['criterion']

    for i in range(epochs):

        rand_indeces = T.randperm(train_X.size()[0])
        shuffled_X = train_X[rand_indeces, :, :]  # randomize the batches for each epoch

        if train_target.dim() > 2:
            shuffled_targets = train_target[rand_indeces, :, :]
        else:
            shuffled_targets = train_target[rand_indeces]

        # replace with view for faster memory access
        epoch_loss = 0
        for j in range(num_iters):
            batch_X = shuffled_X[j*b_size:(j+1)*b_size, :, :]

            if train_target.dim() > 2:
                batch_target = shuffled_targets[j*b_size:(j+1)*b_size, :, :]
            else:
                batch_target = shuffled_targets[j*b_size:(j+1)*b_size]

            training_model.train()
            optimizer.zero_grad()
            batch_prediction = training_model.forward(batch_X)
            loss = criterion(batch_target, batch_prediction)
            loss.backward()
            optimizer.step()

            with T.no_grad():  # add loss to accumulated loss
                epoch_loss += loss.cpu()/(num_iters)  # current epoch loss per batch

        epoch_losses.append(epoch_loss)  # append to epoch losses list for later plotting

        if (i) % n_e_info == 0:  # print stats
            with T.no_grad():
                eval_prediction = training_model.eval().forward(eval_X)  # estimate of eval_X
                eval_loss = criterion(eval_target, eval_prediction)  # loss averaged over batches
            print("Epoch {}/{}, train loss: {:.4f}, eval loss: {:.4f} , finished after {} minutes".format(i, params["epochs"],
                    epoch_loss, eval_loss, int((time()-t_start)//60)))
            if (i) % (50*n_e_info) == 0:  # checkpoint every 10 stat infos
                save_model(training_model, eval_loss, params, i)

        if msvcrt.kbhit():  # handle quitting
            c = msvcrt.getch().decode('utf-8')
            if c == 'q':  # quit
                print('Canceling training process')
                break

    if plot:
       plot_losses(epoch_losses)

    return training_model


params = {"epochs": 60000, "batchsize": 2048, "lr": 0.0005, "weight_decay": 0.0001, 'epoch_0': 0,
            'n_e_info': 10, 'n_sensors': 8, 'n_timesteps': 40, 'target_folder': 'trained_models/Classifier',
            'criterion': ''}

if __name__ == '__main__':

    data_values, data_labels, data_timestamps = load_data('DataCSV.csv')
    room_temps = data_values[:, room_temp_columns]  # just the room temperatures in C
    X = create_batches(room_temps[:, :params['n_sensors']], params['n_timesteps'])

    # train_X, train_targets = prepare_for_AE(X[:-2000, :, :].cuda())  # AE training data
    # eval_X, eval_targets = prepare_for_AE(X[-2000:-500, :, :].cuda())  # AE evaluation data
    # ver_X = X[-500:, :, :].cuda()  # AE verification data


    # prepare data for classifier training
    good_x = X[:-2000, :, :]  # classifier training data
    good_labels = T.ones(good_x.size(0), 1)
    faulty_x = add_linear_error(good_x)
    faulty_labels = T.zeros(faulty_x.size(0), 1)
    train_X = T.cat((good_x, faulty_x), 0)
    train_labels = T.cat((good_labels, faulty_labels), 0)

    eval_X = X[-2000:-500, :, :]  # classifier evaluation data
    eval_labels = T.ones(eval_X.size(0), 1)
    eval_X = T.cat((eval_X, add_linear_error(eval_X)), 0)
    eval_labels = T.cat((eval_labels, T.zeros(eval_labels.size(0), 1)), 0)

    ver_X = X[-500:, :, :]  # classifier verification data
    ver_labels = T.ones(ver_X.size(0), 1)
    ver_X = T.cat((ver_X, add_outliers(add_linear_error(ver_X))), 0)
    ver_labels = T.cat((ver_labels, T.zeros(ver_labels.size(0), 1)), 0)

    model = Classifier(params['n_sensors'])  # load model
    print(model)
    model, checkpoint = load_model(model, 'trained_models/Classifier/Classifier_sen8_ts40_iter039000')
    params['epoch_0'] = checkpoint['epoch']

    params['criterion'] = T.nn.MSELoss()
    device = 'cuda'

    eval_classifier(model, train_X, train_labels)
    eval_classifier(model, eval_X, eval_labels)
    randids = T.randperm(ver_X.size(0))
    eval_classifier(model, ver_X[randids, :, :], ver_labels[randids, :])
    model = train(model.to(device), train_X.to(device), train_labels.to(device), eval_X.to(device), eval_labels.to(device), params)
    eval_classifier(model, train_X, train_labels)
    eval_classifier(model, eval_X, eval_labels)

    # visualize_AE(model, train_X, add_linear_error(train_X.cpu()).cuda(), params)
    # visualize_AE(model, eval_X, add_linear_error(eval_X.cpu()).cuda(), params)
    # params['criterion'] = T.nn.MSELoss()
    # model = train(model, train_X, train_targets, eval_X, eval_targets, params)
    # visualize_AE(model, train_X, add_linear_error(train_X.cpu()).cuda(), params)
    # visualize_AE(model, add_linear_error(eval_X.cpu()).cuda(), params)


