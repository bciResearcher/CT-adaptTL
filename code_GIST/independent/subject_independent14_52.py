import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from os.path import join as pjoin
from SHALLOW import deep

# GIST data path
dfile = h5py.File('../process/DATA52/ku_mi_smt.h5', 'r')

# pretrained model path
outpath = '../pretrain/pretrain14_52'

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


acc = np.zeros([52])
for n in range(52):
    X, Y = get_data(n+1)
    # print(X.shape, Y.shape)

    T_X_train, T_Y_train = X[:100], Y[:100]
    T_X_test, T_Y_test = X[100:], Y[100:]

    T_X_train = T_X_train.transpose([0, 2, 1])
    T_X_test = T_X_test.transpose([0, 2, 1])

    T_X_train = torch.tensor(np.expand_dims(T_X_train, axis=1), dtype=torch.float32)
    T_X_test = torch.tensor(np.expand_dims(T_X_test, axis=1), dtype=torch.float32)
    T_Y_train = torch.tensor(T_Y_train, dtype=torch.long)
    T_Y_test = torch.tensor(T_Y_test, dtype=torch.long)

    T_X_train, T_Y_train = T_X_train.to(device), T_Y_train.to(device)
    T_X_test, T_Y_test = T_X_test.to(device), T_Y_test.to(device)

    target_set = TensorDataset(T_X_train, T_Y_train)
    target_set2 = TensorDataset(T_X_test, T_Y_test)
    target_loader = DataLoader(dataset=target_set, batch_size=40, shuffle=True)
    target_loader2 = DataLoader(dataset=target_set2, batch_size=40, shuffle=True)

    model = deep()
    checkpoint = torch.load(pjoin(outpath, 'model_cv{}.pt'.format(3)))
    model.load_state_dict(checkpoint)
    model.to(device)
    acc[n] = evaluate(model, T_X_test, T_Y_test)
print(np.mean(acc))
print(acc)


