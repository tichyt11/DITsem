from preprocessing import load_data, room_temp_columns, outside_temp_column, prepare_for_training, prepare_faulty_data
from models import LSTMAE, LSTMAE_new, TRAE, LSTMAE_01
from visualization import visualize, visualize_n, visualize_folder, plot_losses, plot_histograms
from utils import load_model, save_model
import torch as T
from time import time
from datetime import datetime
import msvcrt


def train_AE(AE, train_X, train_Y, eval_X, eval_Y, params, plot=True):
    b_size = params['batchsize']
    epochs = params['epochs']
    n_e_info = params['n_e_info']

    num_batches = train_X.size()[0]
    num_iters = num_batches//b_size

    print('Starting training model {} at {}'.format(model.__class__.__name__, datetime.now().time()))
    t_start = time()
    epoch_losses = []

    optimizer = T.optim.Adam(AE.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], eps=1e-4)
    criterion = T.nn.MSELoss()  # averaged squared error

    for i in range(epochs):

        rand_indeces = T.randperm(train_X.size()[0])
        shuffled_X = train_X[rand_indeces, :, :]  # randomize the batches for each epoch

        # replace with view for faster memory access
        epoch_loss = 0
        for j in range(num_iters):
            batch_X = shuffled_X[j*b_size:(j+1)*b_size, :, :]

            AE.train()
            optimizer.zero_grad()
            batch_Y = AE.forward(batch_X)
            loss = criterion(T.unsqueeze(batch_X[:, :, 0], 2), batch_Y)
            loss.backward()
            optimizer.step()

            with T.no_grad():  # add loss to accumulated loss
                epoch_loss += loss.cpu()/(num_iters)  # current epoch loss per batch

        epoch_losses.append(epoch_loss)  # append to epoch losses list for later plotting

        if (i+1) % n_e_info == 0:  # print stats
            with T.no_grad():
                eval_Y = model.eval().forward(eval_X)  # estimate of eval_X
                eval_loss = criterion(T.unsqueeze(eval_X[:, :, 0], 2), eval_Y)  # loss averaged over batches
            print("Epoch {}/{}, train loss: {:.4f}, eval loss: {:.4f} , finished after {} minutes".format(i+1, params["epochs"],
                    epoch_loss, eval_loss, int((time()-t_start)//60)))
            if (i+1) % (10*n_e_info) == 0:  # checkpoint every 10 stat infos
                save_model(model, eval_loss, params, i + 1)

        if msvcrt.kbhit():  # handle quitting
            c = msvcrt.getch().decode('utf-8')
            if c == 'q':  # quit
                print('Canceling training process')
                break

    if plot:
       plot_losses(epoch_losses)

    return AE


params = {"epochs": 30000, "batchsize": 258, "lr": 0.007, "weight_decay": 0.0001, 'epoch_0': 0,
            'n_e_info': 50, 'n_sensors': 8, 'n_timesteps': 40, 'target_folder': 'trained_models/TRAE'}

if __name__ == '__main__':

    data_values, data_labels, data_timestamps = load_data('DataCSV.csv')
    room_temps = data_values[:, room_temp_columns]  # just the room temperatures in C

    # leave the last
    X, Y = prepare_for_training(room_temps[:, :params['n_sensors']], params['n_timesteps'])
    train_X = X[:-2000, :, :].cuda()  # training data
    train_Y = T.unsqueeze(train_X[:, :, 0], 2)
    eval_X = X[-2000:-500, :, :].cuda()  # evaluation data
    eval_Y = T.unsqueeze(eval_X[:, :, 0], 2)  # first sensors values
    ver_X = X[-500:, :, :].cuda()  # verification data
    # trainX_T = trainX_T/T.abs(trainX_T).max()  # scale down by max absolute value

    model = LSTMAE_01(params['n_sensors']).cuda()

    model, checkpoint = load_model(model, 'trained_models/LSTMAE_01/LSTMAE_01_sen8_ts40_iter020000')
    params['epoch_0'] = checkpoint['epoch']

    plot_histograms(model, train_X, prepare_faulty_data(train_X.cpu()).cuda())

    visualize_n(model, ver_X, params)
    visualize_n(model, prepare_faulty_data(ver_X.cpu()).cuda(), params)
    # model = train_AE(model, train_X, train_Y, eval_X, eval_Y, params)
    # visualize(model, ver_X, params)
    # visualize(model, prepare_faulty_data(ver_X.cpu()).cuda(), params)


