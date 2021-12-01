from preprocessing import load_data, room_temp_columns, outside_temp_column, prepare_for_training
from models import LSTMAE, LSTMAE2, LSTMAE0
import torch as T
import matplotlib.pyplot as plt
import os
from time import time
from datetime import datetime
import keyboard


# TODO: better vis, same color for a sensor, line for real data
# TODO: implement a function, which loop through trained models and shows prediction results
# TODO: in training save intermediate losses and plot them at the end of training
# TODO: test model trained on some seq length on larger/smaller time sequences, if it works

def load_model(model, path):
    checkpoint = T.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded model from {}'.format(path))
    return model


def save_model(model, loss, params, epoch):
    sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "trained_models\{}_sen{}_ts{}_iter{}".format(model.__class__.__name__, params['n_sensors'],
                                                                     params['n_timesteps'], epoch))
    T.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'n_sensors': params['n_sensors'],
        'n_timesteps': params['n_timesteps']
    }, sdir)
    print("Saved checkpoint at {} with params {}".format(sdir, params))


def vis(model, trainX_T, params):
    test_x = trainX_T[:3, :, :]
    test_y = model.forward(test_x)

    for i in range(params['n_sensors']):
        plt.scatter(range(params['n_timesteps']), test_x[-1, :, i].cpu().detach().numpy(), label='in %i' % (i+1))
        plt.scatter(range(params['n_timesteps']), test_y[-1, :, i].cpu().detach().numpy(), label='out %i' % (i+1))
    plt.legend()
    plt.show()


def train_AE(trainX_T, AE, params):
    b_size = params['batchsize']
    epochs = params['epochs']
    n_e_info = params['n_e_info']

    num_batches = trainX_T.size()[0]
    num_iters = num_batches//b_size

    print('Starting training model {} at {}'.format(model.__class__.__name__, datetime.now().time()))
    t_start = time()
    total_loss = 0
    epoch_losses = []

    optimizer = T.optim.Adam(AE.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], eps=1e-4)
    criterion = T.nn.MSELoss()

    for i in range(epochs):

        if keyboard.read_key() == 'q':
            print('Canceling training process')
            break

        shuffled_X = trainX_T[T.randperm(trainX_T.size()[0]), :, :]  # randomize the batches
        # replace with view for faster memory access
        epoch_loss = 0

        for j in range(num_iters):
            batch_X = shuffled_X[j*b_size:(j+1)*b_size, :, :]

            optimizer.zero_grad()
            batch_Y = AE.forward(batch_X)
            loss = criterion(batch_X, batch_Y)
            loss.backward()
            optimizer.step()

            with T.no_grad():  # add loss to accumulated loss
                epoch_loss += loss.cpu()/(num_iters*b_size)  # current epoch loss per batch

        total_loss += epoch_loss
        epoch_losses.append(epoch_loss)  # append to epoch losses list for later plotting

        if ((i+1) % n_e_info == 0):  # print stats
            print("Epoch {}/{}, average sequence loss: {} , finished after {} minutes".format(i+1, params["epochs"],
                                                                                     total_loss/(n_e_info),
                                                                                     int((time()-t_start)//60)))
            total_loss = 0
        if (i+1) % 1000 == 0:
            save_model(model, epoch_loss, params, i + 1)

    return AE, epoch_losses


if __name__=='__main__':

    data_values, data_labels, data_timestamps = load_data('DataCSV.csv')
    room_temps = data_values[:, room_temp_columns]  # just the room temperatures in C

    params = {"epochs": 20000, "batchsize": 300, "lr": 0.007, "weight_decay": 0.0001, 'ID': 'ThirdTraining',
              'n_e_info': 100, 'n_sensors': 8, 'n_timesteps': 60}

    trainX_T = prepare_for_training(room_temps[:, :params['n_sensors']], params['n_timesteps']).cuda()
    # trainX_T = trainX_T/T.abs(trainX_T).max()  # scale down by max absolute value

    model = LSTMAE(params['n_sensors']).cuda()
    model = load_model(model, 'trained_models\LSTMAE_ThirdTraining_iter7000')
    vis(model, trainX_T, params)

    # model = train_AE(trainX_T[:2, :, :], model, params)  # try overfitting on first two examples
    model, epoch_losses = train_AE(trainX_T[:, :, :], model, params)
    vis(model, trainX_T, params)


    #     for i in range(len(room_temp_columns)):
    #         plt.scatter(range(100), room_temps[:100, i])
    #
    #     plt.scatter(range(100), data_values[:100, outside_temp_column])
    #     plt.show()

