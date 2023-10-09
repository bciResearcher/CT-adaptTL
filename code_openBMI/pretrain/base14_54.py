'''''

pre-training model for openBMI
'''''
import h5py
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn.functional as F
from SHALLOW_54 import deep
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from os.path import join as pjoin
from pytorchtools import EarlyStopping

# processed HGD data path in device
dfile_sor = h5py.File('../process/DATA14_54/KU_mi_smt.h5', 'r')


def get_data_sor(subj):
    dpath = 's' + str(subj) + '/'
    X = dfile_sor[pjoin(dpath, 'X')]
    Y = dfile_sor[pjoin(dpath, 'Y')]
    return np.array(X), np.array(Y)


def get_multi_data_sor(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data_sor(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


# data load

X_HGD, Y_HGD = get_multi_data_sor(np.arange(1, 15, 1))   #X_HGD 命名不好！6742要打印检查，然后再注释掉！  √
# print(X_HGD.shape)
# pretrained model save path

outpath = './pretrain14_54/'     #最终版的目录名字要再三确认


def evaluate(model, x, y):
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=True)
    model.eval()
    num = 0
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            test_input, test_lab = d
            out = model(test_input)
            _, indices = torch.max(out, dim=1)
            correct = torch.sum(indices == test_lab)
            num += correct
    return num * 1.0 / x.shape[0]


kf = KFold(n_splits=6, shuffle=True)
device = torch.device('cuda')

# How long to wait after last time validation loss improved
patience = 20

train_loss2 = np.zeros([6, 600])
test_loss2 = np.zeros([6, 600])
train_accuracy2 = np.zeros([6, 600])
test_accuracy2 = np.zeros([6, 600])

# Minimum validation loss for each fold
cv_loss = np.ones([6])
# Training process data recording:fold, test acc, validation loss
result = pd.DataFrame(columns=('cv_index', 'test_acc', 'loss'))
np.set_printoptions(threshold=np.inf)

# Training
for cv_index, (train_index, test_index) in enumerate(kf.split(X_HGD)):
    tra_acc = np.zeros([600])
    tra_loss = np.zeros([600])
    tst_acc = np.zeros([600])
    tst_loss = np.zeros([600])

    X_train = X_HGD[train_index]
    Y_train = Y_HGD[train_index]
    X_test = X_HGD[test_index]
    Y_test = Y_HGD[test_index]

    X_train = X_train.transpose([0, 2, 1])
    X_test = X_test.transpose([0, 2, 1])

    X_train = torch.tensor(np.expand_dims(X_train, axis=1), dtype=torch.float32)
    X_test = torch.tensor(np.expand_dims(X_test, axis=1), dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    train_set = TensorDataset(X_train, Y_train)
    test_set = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

    model = deep().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1*0.001, weight_decay=0.5*0.001)

    train_losses = []  # training loss list for each  epoch
    test_losses = []  # validation loss list for each  epoch

    early_stop = EarlyStopping(patience, delta=0.0001, path=pjoin(outpath, 'model_cv{}.pt'.format(cv_index)), verbose=True)

    for epoch in tqdm(range(600)):
        model.train()
        t = 0
        for i, (train_fea, train_lab) in enumerate(train_loader):
            out = model(train_fea)
            _, pred = torch.max(out, dim=1)
            t += (pred == train_lab).sum().cpu().numpy()
            loss = F.nll_loss(out, train_lab)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print('\n Train, epoch: {}, i: {}, loss: {}'.format(epoch, i, loss))
        e = 0
        model.eval()
        for data, target in test_loader:
            output = model(data)
            _, pred = torch.max(output, dim=1)
            e += (pred == target).sum().cpu().numpy()
            loss = F.nll_loss(output, target)
            test_losses.append(loss.item())

        train_loss = np.average(train_losses)
        test_loss = np.average(test_losses)

        tra_loss[epoch] = train_loss
        tst_loss[epoch] = test_loss
        tra_acc[epoch] = t / Y_train.shape[0]
        tst_acc[epoch] = e / Y_test.shape[0]

        if test_loss < cv_loss[cv_index]:
            cv_loss[cv_index] = test_loss  # get the minimum validation loss in this fold

        train_losses = []  # update training loss list for each epoch
        test_losses = []  # update validation loss list for each epoch

        test_acc = evaluate(model, X_test, Y_test)
        print('\n Test: acc: {}'.format(test_acc))

        res = {'cv_index': cv_index, 'test_acc': test_acc.item(), 'loss': test_loss}
        result = result.append(res, ignore_index=True)

        # determine whether the early stopping condition is reached and save the model
        early_stop(test_loss, model)
        if early_stop.early_stop:
            print('Early stopping')
            break

    train_loss2[cv_index] = tra_loss
    test_loss2[cv_index] = tst_loss
    train_accuracy2[cv_index] = tra_acc
    test_accuracy2[cv_index] = tst_acc

# the number of folds with minimum validation loss
min_index = np.argmin(cv_loss, axis=0)
result.to_csv(pjoin(outpath, 'result_cv{}.csv'.format(min_index)), index=False)