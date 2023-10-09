import h5py
import numpy as np
import torch
import shap
from torch.utils.data import TensorDataset, DataLoader
from os.path import join as pjoin
from SHALLOW_54 import deep

# openBMI data path
dfile = h5py.File('../process/DATA54/ku_mi_smt.h5', 'r')
# fine-tuned model path
outpath = '../adapt/model_54_/'
device = torch.device('cuda')


def evaluate(model, x, y):
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=100, shuffle=True)
    model.eval()
    num = 0
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            test_input, test_lab = d
            out = model(test_input)
            _, indices = torch.max(out, dim=1)
            correct = torch.sum(indices == test_lab)
            num += correct.cpu().numpy()
    return num * 1.0 / x.shape[0]


def get_data(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)


# shap values for two classes corresponding to 54 subjects
f_shap_values = []

# samples for explain
f_explain = []
class_names = {'0': ['right'], '1': ['left']}
for t in range(54):
    X, Y = get_data(t+1)

    T_X_test, T_Y_test = X[300:], Y[300:]   # evaluation data

    X_test = T_X_test.transpose([0, 2, 1])

    X_test = torch.tensor(np.expand_dims(X_test, axis=1), dtype=torch.float32)
    Y_test = torch.tensor(T_Y_test, dtype=torch.long)

    X_test, Y_test = X_test.to(device), Y_test.to(device)
    print("X_test:{}".format(X_test.shape))

    # load fine-tuned model for each subject
    model = deep()
    checkpoint = torch.load(pjoin(outpath, 'model_cv{}.pt'.format(t)))
    model.load_state_dict(checkpoint)
    model.to(device)

    # samples for explain
    to_explain = X_test[:]
    print("to_explain:{}".format(to_explain.shape))

    e = shap.GradientExplainer(model, X_test)

    shap_values, indexes = e.shap_values(to_explain, ranked_outputs=1, nsamples=32)
    print("shap_values[0]:{}".format(shap_values[0].shape))
    print("indexes:{}".format(indexes.shape))

    to_explain = np.array(to_explain.cpu())
    indexes = np.array(indexes.cpu())

    shap_values = [np.swapaxes(s, 1, -1) for s in shap_values]
    print("shap_values[0]:{}".format(shap_values[0].shape))

    to_explain = to_explain.swapaxes(-1, 1)
    print("to_explain:{}".format(to_explain.shape))

    shap_zero = np.zeros([np.sum(indexes == 0), 20, 2000, 1])  # right
    shap_one = np.zeros([np.sum(indexes == 1), 20, 2000, 1])  # left

    # group together shap values that belong to the same class and average them
    m, n = 0, 0
    for i in range(indexes.shape[0]):
        if indexes[i] == 0:
            shap_zero[m] = shap_values[0][i]
            m = m + 1
        else:
            shap_one[n] = shap_values[0][i]
            n = n + 1
    shap_zero = sum(shap_zero[:]) / len(shap_zero[:])
    shap_one = sum(shap_one[:]) / len(shap_one[:])
    print("shap_zero:{}".format(shap_zero.shape))

    fin_explain = np.zeros([2, 20, 2000, 1])
    for i in range(2):
        fin_explain[i] = to_explain[list(Y_test).index(i)]
    f_explain.append(fin_explain)

    # the average shap values of two classes are compressed in the time dimension
    fin_shap_values = []
    shap_values = np.zeros([2, 20, 50, 1])
    for i in range(50):
        shap_values[0, :, i, :] = np.mean(shap_zero[:, 40*i:40*(i+1), :], 1)
        shap_values[1, :, i, :] = np.mean(shap_one[:, 40*i:40*(i+1), :], 1)
    fin_shap_values.append(shap_values)
    f_shap_values.append(fin_shap_values)

print(f_shap_values[0][0].shape)

# average shap values of all subjects
f = []
shap_values = np.zeros([2, 20, 50, 1])
for i in range(54):
    shap_values += f_shap_values[i][0]
shap_values = shap_values / 54
f.append(shap_values)

# np.save('54_adjust_shapvalue_2',shap_values)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][0])(torch.tensor([[0], [1]], dtype=torch.long))  # right left
# print(index_names)

shap.image_plot(f, f_explain[0], index_names)
