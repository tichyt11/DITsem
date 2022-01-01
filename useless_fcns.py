# def eval_classifier(model, X, labels):
#     with T.no_grad():
#         prediction = model.eval().forward(X)
#     positives_pred = prediction[labels == 1]
#     true_positives = 100*T.sum(positives_pred >= 0.5)/positives_pred.size(0)
#     false_negatives = 100*T.sum(positives_pred < 0.5)/positives_pred.size(0)
#
#     negatives_pred = prediction[labels == 0]
#     true_negatives = 100*T.sum(negatives_pred < 0.5)/negatives_pred.size(0)
#     false_positives = 100*T.sum(negatives_pred >= 0.5)/negatives_pred.size(0)
#
#     print('Confusion matrix: ')
#     print('     PP    PN')
#     print('P [{:.2f}, {:.2f}]\nN [{:.2f}, {:.2f}]'.format(true_positives, false_negatives, false_positives, true_negatives))
#
#     errors = T.abs(prediction - labels)
#     print('average error: {:.2f}'.format(T.mean(errors)))
#
#
# def visualize_n(model, X, params):
#     k = randint(0, X.size(0)-3)
#     test_x = X[k:k+2, :, :]  # take a random sample from X
#     with T.no_grad():
#         test_y = model.eval().forward(test_x)  # push it through the model
#
#     cmp = plt.cm.get_cmap('gist_rainbow', params['n_sensors'])
#
#     for i in range(params['n_sensors']):
#         plt.plot(range(params['n_timesteps']), test_x[-1, :, i].cpu().detach().numpy(), label='in %i' % (i+1), color=cmp(i))
#         plt.scatter(range(params['n_timesteps']), test_y[-1, :, i].cpu().detach().numpy(), label='out %i' % (i+1), color=cmp(i))
#     plt.legend()
#     plt.show()
#
#
# def visualize_folder_n(model, folder, X, params, labels=None):
#     # given a folder filled with params of the given model, loop through these and show reconstruction for each one
#     k = randint(0, X.size(0)-3)
#     test_x = X[k:k+2, :, :]  # take a random sample from X
#     cmp = plt.cm.get_cmap('gist_rainbow', params['n_sensors'])
#
#     criterion = T.nn.MSELoss()
#
#     if labels == None:
#         labels = [('in %i' % (i+1), 'out %i' % (i+1)) for i in range(['n_sensors'])]
#
#     fnames = os.listdir(folder)  # there must be only params in the folder
#     for fname in fnames:
#         model, checkpoint = load_model(model, os.path.join(folder, fname))
#         with T.no_grad():
#             test_y = model.eval().forward(test_x)  # push it through the model
#
#         loss = criterion(test_x, test_y)
#
#         for i in range(params['n_sensors']):
#             plt.plot(range(params['n_timesteps']), test_x[-1, :, i].cpu().detach().numpy(), label=labels[i][0], color=cmp(i))
#             plt.scatter(range(params['n_timesteps']), test_y[-1, :, i].cpu().detach().numpy(), label=labels[i][1], color=cmp(i))
#         plt.xlabel('t [h]')
#         plt.title('prediction results of {} with loss {}'.format(fname, loss))
#         plt.legend()
#         plt.show()


# def visualize_AE(model, X, faulty_X, params):
#     if X.size(0) > 3:
#         k = randint(0, X.size(0)-3)
#         test_x = X[k:k + 2, :, :]  # take a random sample from X
#         faulty_test_x = faulty_X[k:k + 2, :, :]
#     else:
#         test_x = X
#         faulty_test_x = faulty_X
#
#     with T.no_grad():
#         test_y = model.eval().forward(test_x)  # push it through the model
#         faulty_test_y = model.eval().forward(faulty_test_x)
#
#     plt.plot(range(params['n_timesteps']), test_x[-1, :, 0].cpu().detach().numpy(), label='in', color='red')
#     plt.scatter(range(params['n_timesteps']), test_y[-1, :, 0].cpu().detach().numpy(), label='out', color='red')
#     plt.legend()
#     plt.show()
#
#     plt.plot(range(params['n_timesteps']), faulty_test_x[-1, :, 0].cpu().detach().numpy(), label='in_faulty', color='red')
#     plt.scatter(range(params['n_timesteps']), faulty_test_y[-1, :, 0].cpu().detach().numpy(), label='out_faulty', color='red')
#     plt.legend()
#     plt.show()
#
#     plot_histograms_AE(model, X, faulty_X)
#
#
# def visualize_folder_AE(model, folder, X, params, labels=None):
#     # given a folder filled with params of the given model, loop through these and show reconstruction for each one
#     k = randint(0, X.size(0)-3)
#     test_x = X[k:k+2, :, :]  # take a random sample from X
#     faulty_x = add_outliers(test_x)
#     all_faulty_x = add_outliers(X)
#     criterion = T.nn.MSELoss()
#
#     if labels == None:
#         labels = [('in %i' % (i+1), 'out %i' % (i+1)) for i in range(['n_sensors'])]
#
#     fig, (ax1, ax2, ax3) = plt.subplots(3,1)
#     fnames = os.listdir(folder)  # there must be only params in the folder
#     for fname in fnames:
#         model, checkpoint = load_model(model, os.path.join(folder, fname))
#         with T.no_grad():
#             test_y = model.eval().forward(test_x)  # push it through the model
#             loss = criterion(T.unsqueeze(test_x[:, :, 0], 2), test_y)
#
#             faulty_y = model.eval().forward(faulty_x)
#             faulty_loss = criterion(T.unsqueeze(faulty_x[:, :, 0], 2), faulty_y)
#
#         ax1.clear()
#         ax2.clear()
#         ax3.clear()
#
#         ax1.plot(range(params['n_timesteps']), test_x[-1, :, 0].cpu().detach().numpy(), label=labels[0][0], color='red')
#         ax1.scatter(range(params['n_timesteps']), test_y[-1, :, 0].cpu().detach().numpy(), label=labels[0][1], color='red')
#         ax1.set_title('Faulty - epoch: {} loss: {:.5f}'.format(checkpoint['epoch'], loss.numpy()))
#         ax1.legend()
#
#         ax2.plot(range(params['n_timesteps']), faulty_x[-1, :, 0].cpu().detach().numpy(), label=labels[0][0], color='red')
#         ax2.scatter(range(params['n_timesteps']), faulty_y[-1, :, 0].cpu().detach().numpy(), label=labels[0][1],
#                     color='red')
#         ax2.set_title('Healthy - epoch: {} loss: {:.5f}'.format(checkpoint['epoch'], faulty_loss.numpy()))
#         ax2.legend()
#
#         healthy_real = X[:, :, 0]  # healthy real data
#         faulty_real = all_faulty_x[:, :, 0]  # faulty real data
#         with T.no_grad():
#             healthy_pred = model.eval().forward(X)[:, :, 0]  # prediction from healthy data
#             faulty_pred = model.eval().forward(all_faulty_x)[:, :, 0]  # prediction from faulty data
#         healthy_logSE = T.log10(
#             T.sum(T.square(healthy_pred - healthy_real), 1)).cpu().numpy()  # sum of squared differences
#         faulty_logSE = T.log10(T.sum(T.square(faulty_pred - faulty_real), 1)).cpu().numpy()
#         ax3.hist(np.concatenate((healthy_logSE[None], faulty_logSE[None]), 0).T, bins=100, density=True, cumulative=True,
#                  label=['Healthy data errors', 'Faulty data errors'])
#         ax3.legend()
#         ax3.set_title('Histogram of errors')
#         plt.setp(ax3, xlabel='log10 SSE')
#
#         plt.show(block=False)
#         plt.pause(0.05)

# def plot_histograms_AE(AE, healthy_X, faulty_X):
#     healthy_real = healthy_X[:, :, 0]  # healthy real data
#     faulty_real = faulty_X[:, :, 0]  # faulty real data
#
#     with T.no_grad():
#         healthy_pred = AE.eval().forward(healthy_X)[:, :, 0]  # prediction from healthy data
#         faulty_pred = AE.eval().forward(faulty_X)[:, :, 0]  # prediction from faulty data
#
#     healthy_logSE = T.log(T.sum(T.square(healthy_pred - healthy_real), 1)).cpu().numpy()  # sum of squared differences
#     faulty_logSE = T.log(T.sum(T.square(faulty_pred - faulty_real), 1)).cpu().numpy()
#
#     plt.hist(np.concatenate((healthy_logSE[None], faulty_logSE[None]), 0).T, bins=100,
#              label=['Healthy data prediction errors', 'Faulty data prediction errors'])
#     plt.legend()
#     plt.show()
#
#     plt.hist(np.concatenate((healthy_logSE[None], faulty_logSE[None]), 0).T, bins=100, cumulative=True, density=True,
#              label=['Healthy data prediction errors', 'Faulty data prediction errors'])
#     plt.legend()
#     plt.show()


# def prepare_for_AE(data_values):
#     X = data_values  # training data
#     targets = T.unsqueeze(X[:, :, 0], 2)  # take first sensors as target
#     return X, targets
#
#
# def prepare_for_classifier(data_values):
#     X = data_values  # training data
#     p = X.size(0)//3  # partitions
#     lin_X = add_linear_error(X[:p, :, :])
#     outlier_X = add_outliers(X[p:2*p, :, :])
#     offset_X = add_offset_error(X[2*p:, :, :])
#     targets = T.cat((T.ones(X.size(0), 1), T.zeros(X.size(0), 1)), 0).float()
#     X = T.cat((X, lin_X, outlier_X, offset_X), 0)
#     return X, targets