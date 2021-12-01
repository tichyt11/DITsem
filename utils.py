import torch as T
import os


def load_model(model, path):
    checkpoint = T.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded model from {}'.format(path))
    return model, checkpoint


def save_model(model, loss, params, epoch):
    sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "{}/{}_sen{}_ts{}_iter{:06d}".format(params['target_folder'], model.__class__.__name__,
                                                          params['n_sensors'],
                                                          params['n_timesteps'], (epoch + params['epoch_0']) ))
    T.save({
        'epoch': (epoch + params['epoch_0']),
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'n_sensors': params['n_sensors'],
        'n_timesteps': params['n_timesteps']
    }, sdir)
    print("Saved checkpoint at {} with params {}".format(sdir, params))
