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