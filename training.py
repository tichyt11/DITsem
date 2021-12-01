from preprocessing import load_data, room_temp_columns, outside_temp_column, prepare_for_training
from models import LSTMAE, LSTMAE2, LSTMAE0
from visualization import visualize, visualize_folder, plot_losses
from utils import load_model, save_model
import torch as T
from time import time
from datetime import datetime
import msvcrt

# TODO: test model trained on some seq length on larger/smaller time sequences, if it works
# TODO: more effiecient way to shuffle dataset?

def train_AE(train_X, eval_X, AE, params, plot=True):
    b_size = params['batchsize']
    epochs = params['epochs']
    n_e_info = params['n_e_info']

    num_batches = train_X.size()[0]
    num_iters = num_batches//b_size

    print('Starting training model {} at {}'.format(model.__class__.__name__, datetime.now().time()))
    t_start = time()
    epoch_losses = []

    optimizer = T.optim.Adam(AE.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], eps=1e-4)
    criterion = T.nn.MSELoss()

    for i in range(epochs):

        shuffled_X = train_X[T.randperm(train_X.size()[0]), :, :]  # randomize the batches
        # replace with view for faster memory access
        epoch_loss = 0
        for j in range(num_iters):
            batch_X = shuffled_X[j*b_size:(j+1)*b_size, :, :]

            AE.train()
            optimizer.zero_grad()
            batch_Y = AE.forward(batch_X)
            loss = criterion(batch_X, batch_Y)
            loss.backward()
            optimizer.step()

            with T.no_grad():  # add loss to accumulated loss
                epoch_loss += loss.cpu()/(num_iters*b_size)  # current epoch loss per batch

        epoch_losses.append(epoch_loss)  # append to epoch losses list for later plotting

        if (i+1) % n_e_info == 0:  # print stats
            with T.no_grad():
                eval_Y = model.eval().forward(eval_X)  # estimate of eval_X
                eval_loss = criterion(eval_X, eval_Y)/eval_Y.size(0)  # loss averaged over batches
            print("Epoch {}/{}, average sequence loss: {} , finished after {} minutes".format(i+1, params["epochs"],
                    eval_loss, int((time()-t_start)//60)))
            if (i+1) % (10*n_e_info) == 0:  # checkpoint every 10 stat infos
                save_model(model, eval_loss, params, i + 1)

        if msvcrt.kbhit():  # handle quitting
            if msvcrt.getch() == 'q':
                print('Canceling training process')
                break

    if plot:
       plot_losses(epoch_losses)

    return AE


params = {"epochs": 20000, "batchsize": 300, "lr": 0.007, "weight_decay": 0.0001, 'epoch_0': 0,
            'n_e_info': 50, 'n_sensors': 8, 'n_timesteps': 40, 'target_folder': 'trained_models/LSTMAE'}

if __name__ == '__main__':

    data_values, data_labels, data_timestamps = load_data('DataCSV.csv')
    room_temps = data_values[:, room_temp_columns]  # just the room temperatures in C

    # leave the last
    X = prepare_for_training(room_temps[:, :params['n_sensors']], params['n_timesteps'])
    train_X = X[:-2000, :, :].cuda()  # training data
    eval_X = X[-2000:-500, :, :].cuda()  # evaluation data
    ver_X = X[-500:, :, :].cuda()  # verification data
    # trainX_T = trainX_T/T.abs(trainX_T).max()  # scale down by max absolute value

    model = LSTMAE(params['n_sensors']).cuda()
    # model, checkpoint = load_model(model, 'trained_models\')
    # params['epoch_0'] = checkpoint['epoch']
    visualize(model, ver_X, params)
    model = train_AE(train_X, eval_X, model, params)
    visualize(model, ver_X, params)

