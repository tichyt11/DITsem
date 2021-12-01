from matplotlib import pyplot as plt
from models import LSTMAE, LSTMAE2
from preprocessing import prepare_for_training, room_temp_columns, load_data


data_values, data_labels, data_timestamps = load_data('DataCSV.csv')
room_temps = data_values[:, room_temp_columns]  # just the room temperatures in C

n_sequences = 1

# TODO: split dataset for validation and evaluation
trainX_T = prepare_for_training(room_temps[:150, :n_sequences], 100)
model = LSTMAE2(n_sequences)

out = model.forward(trainX_T)
print(out.shape)

in_plot = plt.scatter(range(100), trainX_T[0, :, 0].detach().numpy(), label='in')
plt.scatter(range(100), out[0, :, 0].detach().numpy(), label='out')
plt.legend()
plt.show()